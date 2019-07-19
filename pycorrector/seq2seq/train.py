# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief: Train seq2seq model for text grammar error correction
import glob
import os
import re
import sys
import time

import torch
from torch.autograd import Variable

sys.path.append('../..')
from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import PAD_TOKEN, save_word_dict, create_batch_file, process_minibatch_explicit, \
    build_dataset, read_vocab
from pycorrector.utils.logger import logger
from pycorrector.seq2seq.seq2seq_model import Seq2Seq
from pycorrector.seq2seq.eval import eval


def train(train_path=config.train_path,
          output_dir=config.output_dir,
          save_model_dir=config.save_model_dir,
          vocab_path=config.vocab_path,
          val_path=config.val_path,
          vocab_max_size=config.vocab_max_size,
          vocab_min_count=config.vocab_min_count,
          batch_size=config.batch_size,
          epochs=config.epochs,
          learning_rate=0.0001,
          src_emb_dim=128,
          trg_emb_dim=128,
          src_hidden_dim=256,
          trg_hidden_dim=256,
          src_num_layers=1,
          batch_first=True,
          src_bidirection=True,
          dropout=0.0,
          attn_method='luong_concat',
          repetition='vanilla',
          network='lstm',
          pointer_net=True,
          attn_decoder=True,
          shared_embedding=True,
          share_emb_weight=True,
          src_seq_lens=128,
          trg_seq_lens=128,
          grad_clip=2.0,
          save_model_batch_num=config.save_model_batch_num,
          gpu_id=config.gpu_id):
    print('Training model...')

    if gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('device:', device)
    source_texts, target_texts = build_dataset(train_path)
    print('source_texts:', source_texts[0])
    print('target_texts:', target_texts[0])

    vocab2id = read_vocab(source_texts, max_size=vocab_max_size, min_count=vocab_min_count)
    num_encoder_tokens = len(vocab2id)
    max_input_texts_len = max([len(text) for text in source_texts])
    print('num of samples:', len(source_texts))
    print('num of unique input tokens:', num_encoder_tokens)
    print('max sequence length for inputs:', max_input_texts_len)

    id2vocab = {v: k for k, v in vocab2id.items()}
    # save word dict
    save_word_dict(vocab2id, vocab_path)
    print('The vocabulary file:%s, size: %s' % (vocab_path, len(vocab2id)))

    model = Seq2Seq(
        src_emb_dim=src_emb_dim,
        trg_emb_dim=trg_emb_dim,
        src_hidden_dim=src_hidden_dim,
        trg_hidden_dim=trg_hidden_dim,
        src_vocab_size=len(vocab2id),
        trg_vocab_size=len(vocab2id),
        src_nlayer=src_num_layers,
        batch_first=batch_first,
        src_bidirect=src_bidirection,
        dropout=dropout,
        attn_method=attn_method,
        repetition=repetition,
        network=network,
        pointer_net=pointer_net,
        shared_emb=shared_embedding,
        attn_decoder=attn_decoder,
        share_emb_weight=share_emb_weight,
        device=device
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # read the last check point and continue training
    uf_model = [0, -1]
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    model_para_files = glob.glob(os.path.join(save_model_dir, '*.model'))
    if len(model_para_files) > 0:
        uf_model = []
        for fl_ in model_para_files:
            arr = re.split('\/', fl_)[-1]
            arr = re.split('\_|\.', arr)
            uf_model.append([int(arr[1]), int(arr[2])])
        uf_model = sorted(uf_model)[-1]
        fl_ = os.path.join(save_model_dir, 'seq2seq_' + str(uf_model[0]) + '_' + str(uf_model[1]) + '.model')
        model.load_state_dict(torch.load(fl_))

    # train models
    losses = []
    start_time = time.time()
    last_model_path = ''
    model.train()
    for epoch in range(uf_model[0], epochs):
        n_batch = create_batch_file(output_dir, file_type='train', file_path=train_path, batch_size=batch_size)
        print('The number of batches: {}'.format(n_batch))
        for batch_id in range(n_batch):
            ext_id2oov, src_arr, trg_input_arr, src_arr_ex, trg_output_arr_ex = process_minibatch_explicit(
                batch_id=batch_id,
                output_dir=output_dir,
                file_type='train',
                batch_size=batch_size,
                vocab2id=vocab2id,
                max_lens=[src_seq_lens, trg_seq_lens])

            src_var = Variable(torch.LongTensor(src_arr))
            trg_input_var = Variable(torch.LongTensor(trg_input_arr))
            # extend oov
            src_var_ex = Variable(torch.LongTensor(src_arr_ex))
            trg_output_var_ex = Variable(torch.LongTensor(trg_output_arr_ex))

            src_var = src_var.to(device)
            trg_input_var = trg_input_var.to(device)
            src_var_ex = src_var_ex.to(device)
            trg_output_var_ex = trg_output_var_ex.to(device)

            weight_mask = torch.ones(len(vocab2id) + len(ext_id2oov)).to(device)
            weight_mask[vocab2id[PAD_TOKEN]] = 0
            loss_criterion = torch.nn.NLLLoss(weight=weight_mask).to(device)

            logits, attn_, p_gen, loss_cv = model(src_var, trg_input_var)
            logits = torch.softmax(logits, dim=2)
            # use the pointer generator loss
            if len(ext_id2oov) > 0:
                logits = model.cal_dist_explicit(src_var_ex, logits, attn_, p_gen, vocab2id, ext_id2oov)
                logits = logits + 1e-20
            else:
                logits = model.cal_dist(src_var, logits, attn_, p_gen, vocab2id)

            if batch_id % 1 == 0:
                word_prob = logits.topk(1, dim=2)[1].squeeze(2).data.cpu().numpy()

            logits = torch.log(logits)
            loss = loss_criterion(
                logits.contiguous().view(-1, len(vocab2id) + len(ext_id2oov)),
                trg_output_var_ex.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            end_time = time.time()
            losses.append([
                epoch, batch_id,
                loss.data.cpu().numpy(),
                loss_cv.data.cpu().numpy()[0],
                (end_time - start_time) / 3600.0])

            if batch_id % save_model_batch_num == 0:
                model_path = os.path.join(save_model_dir, 'seq2seq_' + str(epoch) + '_' + str(batch_id) + '.model')
                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                    logger.info("Model save to " + model_path)

            if batch_id % 1 == 0:
                end_time = time.time()
                sen_pred = [id2vocab[x] if x in id2vocab else ext_id2oov[x] for x in word_prob[0]]
                print('epoch={}, batch={}, loss={}, loss_cv={}, time_escape={}s={}h'.format(
                    epoch, batch_id,
                    loss.data.cpu().numpy(),
                    loss_cv.data.cpu().numpy()[0],
                    end_time - start_time, (end_time - start_time) / 3600.0
                ))
                print(' '.join(sen_pred))
            del logits, attn_, p_gen, loss_cv, loss

        with open(os.path.join(save_model_dir, 'loss.txt'), 'a', encoding='utf-8') as f:
            for i in losses:
                f.write(str(i) + '\n')
        model_path = os.path.join(save_model_dir, 'seq2seq_' + str(epoch) + '_' + str(batch_id) + '.model')
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
            logger.info("Model save to " + model_path)
            last_model_path = model_path
    logger.info("Training has finished.")

    # Eval model
    eval(model, last_model_path, val_path, output_dir, batch_size, vocab2id, src_seq_lens, trg_seq_lens, device)
    logger.info("Eval has finished.")


if __name__ == "__main__":
    train()

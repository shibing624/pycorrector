# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: train seq2seq model

# #### PyTorch代码
# - [seq2seq-tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# - [Tutorial from Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
# - [IBM seq2seq](https://github.com/IBM/pytorch-seq2seq)
# - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
# - [text-generation](https://github.com/shibing624/text-generation)
"""

import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.append('../..')
from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import gen_examples, load_bert_data
from pycorrector.seq2seq.data_reader import read_vocab, create_dataset, one_hot, save_word_dict, load_word_dict
from pycorrector.seq2seq.seq2seq import Seq2Seq, LanguageModelCriterion
from pycorrector.seq2seq.data_reader import PAD_TOKEN
from pycorrector.seq2seq.convseq2seq import ConvSeq2Seq
from pycorrector.utils.logger import logger
from pycorrector.seq2seq.seq2seq_model import Seq2SeqModel

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_seq2seq_model(model, data, device, loss_fn):
    model.eval()
    total_num_words = 0.
    total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss / total_num_words)


def train_seq2seq_model(model, train_data, device, loss_fn, optimizer, model_dir, epochs=20):
    best_loss = 1e3
    train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_num_words = 0.
        total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # update optimizer
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, epochs, it, len(train_data),
                                                                          loss.item()))
        cur_loss = total_loss / total_num_words
        print("Epoch :{}/{}, Training loss:{:.4f}".format(epoch, epochs, cur_loss))
        if epoch % 1 == 0:
            # find best model
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)
            if is_best:
                model_path = os.path.join(model_dir, 'seq2seq.pth')
                torch.save(model.state_dict(), model_path)
                print('Epoch:{}, save new bert model:{}'.format(epoch, model_path))
            if dev_data:
                evaluate_seq2seq_model(model, dev_data, device, loss_fn)


def evaluate_convseq2seq_model(model, data, device, loss_fn):
    model.eval()
    last_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            src = torch.from_numpy(mb_x).to(device).long()
            trg = torch.from_numpy(mb_y).to(device).long()
            output, attn = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = loss_fn(output, trg)
            last_loss = loss.item()
    print("Evaluation loss:{:.4f}".format(last_loss))


def train_convseq2seq_model(model, train_data, device, loss_fn, optimizer, model_dir, epochs=20):
    print('start train...')
    best_loss = 1e3
    train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        total_iter = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
            src = torch.from_numpy(mb_x).to(device).long()
            trg = torch.from_numpy(mb_y).to(device).long()
            output, attn = model(src, trg[:, :-1])

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = loss_fn(output, trg)
            total_loss += loss.item()
            total_iter += 1

            # update optimizer
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, epochs, it, len(train_data),
                                                                          loss.item()))
        cur_loss = total_loss / total_iter
        print("Epoch :{}/{}, Training loss:{:.4f}".format(epoch, epochs, cur_loss))
        if epoch % 1 == 0:
            # find best model
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)
            if is_best:
                model_path = os.path.join(model_dir, 'convseq2seq.pth')
                torch.save(model.state_dict(), model_path)
                print('Epoch:{}, save new bert model:{}'.format(epoch, model_path))
            if dev_data:
                evaluate_convseq2seq_model(model, dev_data, device, loss_fn)


def train(arch, train_path, batch_size, embed_size, hidden_size, dropout, epochs,
          src_vocab_path, trg_vocab_path, model_dir, max_length, use_segment, model_name_or_path):
    logger.info("device: {}".format(device))
    arch = arch.lower()
    os.makedirs(model_dir, exist_ok=True)
    if arch in ['seq2seq', 'convseq2seq']:
        source_texts, target_texts = create_dataset(train_path, split_on_space=use_segment)

        src_2_ids = read_vocab(source_texts)
        trg_2_ids = read_vocab(target_texts)
        save_word_dict(src_2_ids, src_vocab_path)
        save_word_dict(trg_2_ids, trg_vocab_path)
        src_2_ids = load_word_dict(src_vocab_path)
        trg_2_ids = load_word_dict(trg_vocab_path)

        id_2_srcs = {v: k for k, v in src_2_ids.items()}
        id_2_trgs = {v: k for k, v in trg_2_ids.items()}
        train_src, train_trg = one_hot(source_texts, target_texts, src_2_ids, trg_2_ids, sort_by_len=True)

        logger.debug(f'src: {[id_2_srcs[i] for i in train_src[0]]}')
        logger.debug(f'trg: {[id_2_trgs[i] for i in train_trg[0]]}')

        train_data = gen_examples(train_src, train_trg, batch_size, max_length)

        if arch == 'seq2seq':
            # Normal seq2seq
            model = Seq2Seq(encoder_vocab_size=len(src_2_ids),
                            decoder_vocab_size=len(trg_2_ids),
                            embed_size=embed_size,
                            enc_hidden_size=hidden_size,
                            dec_hidden_size=hidden_size,
                            dropout=dropout).to(device)
            logger.info(model)
            loss_fn = LanguageModelCriterion().to(device)
            optimizer = torch.optim.Adam(model.parameters())

            train_seq2seq_model(model, train_data, device, loss_fn, optimizer, model_dir, epochs=epochs)
        else:
            # Conv seq2seq model
            trg_pad_idx = trg_2_ids[PAD_TOKEN]
            model = ConvSeq2Seq(encoder_vocab_size=len(src_2_ids),
                                decoder_vocab_size=len(trg_2_ids),
                                embed_size=embed_size,
                                enc_hidden_size=hidden_size,
                                dec_hidden_size=hidden_size,
                                dropout=dropout,
                                trg_pad_idx=trg_pad_idx,
                                device=device,
                                max_length=max_length).to(device)
            logger.info(model)
            loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
            optimizer = torch.optim.Adam(model.parameters())

            train_convseq2seq_model(model, train_data, device, loss_fn, optimizer, model_dir, epochs=epochs)
    elif arch == 'bertseq2seq':
        # Bert Seq2seq model
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": max_length if max_length else 128,
            "train_batch_size": batch_size if batch_size else 8,
            "num_train_epochs": epochs if epochs else 10,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "silent": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "best_model_dir": os.path.join(model_dir, 'best_model'),
            "use_multiprocessing": False,
            "save_best_model": True,
            "max_length": max_length if max_length else 128,  # The maximum length of the sequence to be generated.
            "output_dir": model_dir if model_dir else "./output/bertseq2seq/",
        }

        use_cuda = True if torch.cuda.is_available() else False
        # encoder_type=None, encoder_name=None, decoder_name=None
        # encoder_name="bert-base-chinese"
        model = Seq2SeqModel("bert", model_name_or_path, model_name_or_path, args=model_args, use_cuda=use_cuda)

        logger.info('start train bertseq2seq ...')
        data = load_bert_data(train_path, use_segment)
        logger.info(f'load data done, data size: {len(data)}')
        logger.debug(f'data samples: {data[:10]}')
        train_data, dev_data = train_test_split(data, test_size=0.1, shuffle=True)

        train_df = pd.DataFrame(train_data, columns=['input_text', 'target_text'])
        dev_df = pd.DataFrame(dev_data, columns=['input_text', 'target_text'])

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=dev_df, matches=count_matches)
    else:
        logger.error('error arch: {}'.format(arch))
        raise ValueError("Model arch choose error. Must use one of seq2seq model.")


if __name__ == '__main__':
    train(
        config.arch,
        config.train_path,
        config.batch_size,
        config.embed_size,
        config.hidden_size,
        config.dropout,
        config.epochs,
        config.src_vocab_path,
        config.trg_vocab_path,
        config.model_dir,
        config.max_length,
        config.use_segment,
        config.model_name_or_path,
    )

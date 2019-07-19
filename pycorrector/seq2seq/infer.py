# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief:

import os
import sys
import time

import torch
from torch.autograd import Variable

sys.path.append('../..')
from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import create_batch_file, process_minibatch_explicit_test, \
    show_progress, load_word_dict, PAD_TOKEN, END_TOKEN, UNK_TOKEN
from pycorrector.seq2seq.beam_search import fast_beam_search
from pycorrector.seq2seq.seq2seq_model import Seq2Seq
from pycorrector.utils.logger import logger


def infer_by_file(model_path,
                  output_dir,
                  test_path,
                  predict_out_path,
                  vocab_path,
                  src_seq_lens=128,
                  trg_seq_lens=128,
                  beam_size=5,
                  batch_size=1,
                  gpu_id=0):
    if gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('device:', device)
    test_batch = create_batch_file(output_dir, 'test', test_path, batch_size=batch_size)
    print('The number of batches (test): {}'.format(test_batch))

    vocab2id = load_word_dict(vocab_path)
    id2vocab = {v: k for k, v in vocab2id.items()}
    print('The vocabulary file:%s, size: %s' % (vocab_path, len(vocab2id)))

    model = Seq2Seq(
        src_vocab_size=len(vocab2id),
        trg_vocab_size=len(vocab2id),
        src_nlayer=1,
        pointer_net=True,
        shared_emb=True,
        attn_decoder=True,
        share_emb_weight=True,
        device=device).to(device)
    print(model)

    model.eval()
    with torch.no_grad():
        print("Model file {}".format(model_path))
        print("Batch Size = {}, Beam Size = {}".format(batch_size, beam_size))
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            print('Load model to CPU')
            # 把所有的张量加载到CPU中
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        start_time = time.time()
        with open(predict_out_path, 'w', encoding='utf-8') as f:
            for batch_id in range(test_batch):
                ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr = process_minibatch_explicit_test(
                    batch_id=batch_id,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    vocab2id=vocab2id,
                    src_lens=src_seq_lens)

                src_var = Variable(torch.LongTensor(src_var)).to(device)
                src_var_ex = Variable(torch.LongTensor(src_var_ex)).to(device)
                src_msk = Variable(torch.FloatTensor(src_msk)).to(device)

                beam_seq, beam_prb, beam_attn_out = fast_beam_search(
                    model=model,
                    src_text=src_var,
                    src_text_ex=src_var_ex,
                    vocab2id=vocab2id,
                    ext_id2oov=ext_id2oov,
                    beam_size=beam_size,
                    max_len=trg_seq_lens,
                    network='lstm',
                    pointer_net=True,
                    oov_explicit=True,
                    attn_decoder=True,
                    device=device)
                src_msk = src_msk.repeat(1, beam_size).view(src_msk.size(0), beam_size, src_seq_lens).unsqueeze(0)
                # copy unknown words
                beam_attn_out = beam_attn_out * src_msk
                beam_copy = beam_attn_out.topk(1, dim=3)[1].squeeze(-1)
                beam_copy = beam_copy[:, :, 0].transpose(0, 1)
                wdidx_copy = beam_copy.data.cpu().numpy()
                for b in range(len(trg_arr)):
                    arr = []
                    gen_text = beam_seq.data.cpu().numpy()[b, 0]
                    gen_text = [id2vocab[wd] if wd in id2vocab else ext_id2oov[wd] for wd in gen_text]
                    gen_text = gen_text[1:]
                    for j in range(len(gen_text)):
                        if gen_text[j] == UNK_TOKEN:
                            gen_text[j] = src_arr[b][wdidx_copy[b, j]]
                        if gen_text[j] == END_TOKEN:
                            gen_text = gen_text[:j]
                            break
                    arr.append(''.join(gen_text))
                    arr.append(trg_arr[b])
                    f.write(' '.join(arr) + '\n')

                end_time = time.time()
                show_progress(batch_id, test_batch, str((end_time - start_time) / 3600)[:8] + "h")


class Inference:
    def __init__(self, vocab_path='',
                 model_path='',
                 src_seq_lens=128,
                 trg_seq_lens=128,
                 beam_size=5,
                 batch_size=1,
                 gpu_id=0):
        use_gpu = False
        if gpu_id > -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
            if torch.cuda.is_available():
                device = torch.device('cuda')
                use_gpu = True
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        print('device:', device)
        # load vocab
        self.vocab2id = load_word_dict(vocab_path)
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}
        logger.debug('Loaded vocabulary file:%s, size: %s' % (vocab_path, len(self.vocab2id)))

        # load model
        start_time = time.time()
        self.model = self._create_model(self.vocab2id, device)
        if use_gpu:
            self.model.load_state_dict(torch.load(model_path))
        else:
            # 把所有的张量加载到CPU中
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        logger.info("Loaded model:%s, spend:%s s" % (model_path, time.time() - start_time))

        self.model.eval()
        self.src_seq_lens = src_seq_lens
        self.trg_seq_lens = trg_seq_lens
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.device = device

    def _create_model(self, vocab2id, device):
        model = Seq2Seq(
            src_vocab_size=len(vocab2id),
            trg_vocab_size=len(vocab2id),
            src_nlayer=1,
            pointer_net=True,
            shared_emb=True,
            attn_decoder=True,
            share_emb_weight=True,
            device=device
        ).to(device)
        print(model)
        return model

    def _encode_text(self, text):
        ext_vocab = {}
        ext_id2oov = {}

        text = text.strip()
        tokens = list(text)
        for wd in tokens:
            if wd not in self.vocab2id:
                ext_vocab[wd] = {}

        cnt = len(self.vocab2id)
        for wd in ext_vocab:
            ext_vocab[wd] = cnt
            ext_id2oov[cnt] = wd
            cnt += 1

        src_arr = []
        src_idx = []
        src_idx_ex = []
        src_wt = []

        src_arr.append(tokens)
        wrong2id = [self.vocab2id[wd] if wd in self.vocab2id else self.vocab2id[UNK_TOKEN] for wd in tokens]
        src_idx.append(wrong2id)
        wrong2id = [self.vocab2id[wd] if wd in self.vocab2id else ext_vocab[wd] for wd in tokens]
        src_idx_ex.append(wrong2id)
        wrong2wt = [0.0 if wd in self.vocab2id else 1.0 for wd in tokens]
        src_wt.append(wrong2wt)

        src_idx = [itm[:self.src_seq_lens] for itm in src_idx]
        src_var = [itm + [self.vocab2id[PAD_TOKEN]] * (self.src_seq_lens - len(itm)) for itm in src_idx]

        src_idx_ex = [itm[:self.src_seq_lens] for itm in src_idx_ex]
        src_var_ex = [itm + [self.vocab2id[PAD_TOKEN]] * (self.src_seq_lens - len(itm)) for itm in src_idx_ex]

        src_wt = [itm[:self.src_seq_lens] for itm in src_wt]
        src_msk = [itm + [0.0] * (self.src_seq_lens - len(itm)) for itm in src_wt]

        src_arr = [itm[:self.src_seq_lens] for itm in src_arr]
        src_arr = [itm + [PAD_TOKEN] * (self.src_seq_lens - len(itm)) for itm in src_arr]

        # ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr

        src_var = Variable(torch.LongTensor(src_var)).to(self.device)
        src_var_ex = Variable(torch.LongTensor(src_var_ex)).to(self.device)
        src_msk = Variable(torch.FloatTensor(src_msk)).to(self.device)

        return ext_id2oov, src_var, src_var_ex, src_arr, src_msk

    def _beam_search(self, ext_id2oov, src_var, src_var_ex, src_arr, src_msk):
        beam_seq, beam_prb, beam_attn_out = fast_beam_search(
            model=self.model,
            src_text=src_var,
            src_text_ex=src_var_ex,
            vocab2id=self.vocab2id,
            ext_id2oov=ext_id2oov,
            beam_size=self.beam_size,
            max_len=self.trg_seq_lens,
            network='lstm',
            pointer_net=True,
            oov_explicit=True,
            attn_decoder=True,
            device=self.device)
        src_msk = src_msk.repeat(1, self.beam_size) \
            .view(src_msk.size(0), self.beam_size, self.src_seq_lens) \
            .unsqueeze(0)
        # copy unknown words
        beam_attn_out = beam_attn_out * src_msk
        beam_copy = beam_attn_out.topk(1, dim=3)[1].squeeze(-1)
        beam_copy = beam_copy[:, :, 0].transpose(0, 1)
        wdidx_copy = beam_copy.data.cpu().numpy()

        gen_text = beam_seq.data.cpu().numpy()[0, 0]
        gen_text = [self.id2vocab[wd] if wd in self.id2vocab else ext_id2oov[wd] for wd in gen_text]
        gen_text = gen_text[1:]
        for j in range(len(gen_text)):
            if gen_text[j] == UNK_TOKEN:
                gen_text[j] = src_arr[0][wdidx_copy[0, j]]
            if gen_text[j] == END_TOKEN:
                gen_text = gen_text[:j]
                break
            if gen_text[j] == PAD_TOKEN:
                gen_text[j] = ''
        gen_text.insert(0, src_arr[0][0])
        return ''.join(gen_text)

    def infer(self, text):
        ext_id2oov, src_var, src_var_ex, src_arr, src_msk = self._encode_text(text)
        gen_text = self._beam_search(ext_id2oov, src_var, src_var_ex, src_arr, src_msk)
        return gen_text


if __name__ == "__main__":
    inputs = [
        '少先队员因该给老人让坐',
        '少先队员应该给老人让坐',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
    ]
    inference = Inference(vocab_path=config.vocab_path,
                          model_path=config.model_path)
    for i in inputs:
        gen = inference.infer(i)
        print('input:', i, 'output:', gen)

    if not os.path.exists(config.predict_out_path):
        # infer test file
        infer_by_file(model_path=config.model_path,
                      output_dir=config.output_dir,
                      test_path=config.test_path,
                      predict_out_path=config.predict_out_path,
                      vocab_path=config.vocab_path)

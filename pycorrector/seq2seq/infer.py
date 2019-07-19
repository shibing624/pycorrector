# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief:

import sys
import time

import torch
from torch.autograd import Variable

sys.path.append('../..')
from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import create_batch_file, process_minibatch_explicit_test, \
    show_progress, UNK_TOKEN, load_word_dict
from pycorrector.seq2seq.beam_search import fast_beam_search
from pycorrector.seq2seq.seq2seq_model import Seq2Seq


def infer_by_file(model_path,
                  output_dir,
                  test_path,
                  predict_out_path,
                  vocab_path,
                  src_seq_lens=128,
                  trg_seq_lens=128,
                  beam_size=5,
                  batch_size=1,
                  device=torch.device('cpu')):
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
        model.load_state_dict(torch.load(model_path))

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
                    attn_decoder=True)
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
                    arr.append(' '.join(gen_text))
                    arr.append(trg_arr[b])
                    f.write(' '.join(arr) + '\n')

                end_time = time.time()
                show_progress(batch_id, test_batch, str((end_time - start_time) / 3600)[:8] + "h")


if __name__ == "__main__":
    infer_by_file(model_path=config.model_path,
         output_dir=config.output_dir,
         test_path=config.test_path,
         predict_out_path=config.predict_out_path,
         vocab_path=config.vocab_path)
    inputs = [
        '少先队员应该给老人让坐',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
    ]
    for i in inputs:
        inference.infer(i)

    while True:
        input_str = input('input your string:')
        inference.infer(input_str)

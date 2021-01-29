# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

import numpy as np
import torch

sys.path.append('../..')

from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import SOS_TOKEN, EOS_TOKEN
from pycorrector.seq2seq.data_reader import load_word_dict
from pycorrector.seq2seq.model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: %s' % device)


class Inference(object):
    def __init__(self, model_path, src_vocab_path, trg_vocab_path, embed_size=50, hidden_size=50, dropout=0.5):
        self.src_2_ids = load_word_dict(src_vocab_path)
        self.trg_2_ids = load_word_dict(trg_vocab_path)

        encoder = Encoder(vocab_size=len(self.src_2_ids),
                          embed_size=embed_size,
                          enc_hidden_size=hidden_size,
                          dec_hidden_size=hidden_size,
                          dropout=dropout)
        decoder = Decoder(vocab_size=len(self.trg_2_ids),
                          embed_size=embed_size,
                          enc_hidden_size=hidden_size,
                          dec_hidden_size=hidden_size,
                          dropout=dropout)
        self.model = Seq2Seq(encoder, decoder).to(device)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, query):
        id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}

        src_ids = [self.src_2_ids[i] for i in query if i in self.src_2_ids]
        mb_x = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
        mb_x_len = torch.from_numpy(np.array([len(src_ids)])).long().to(device)
        bos = torch.Tensor([[self.trg_2_ids[SOS_TOKEN]]]).long().to(device)

        translation, attn = self.model.translate(mb_x, mb_x_len, bos)
        translation = [id_2_trgs[i] for i in translation.data.cpu().numpy().reshape(-1) if i in id_2_trgs]
        trans = []
        for word in translation:
            if word != EOS_TOKEN:
                trans.append(word)
            else:
                break
        return ''.join(trans)


if __name__ == "__main__":
    m = Inference(config.model_path, config.src_vocab_path, config.trg_vocab_path, embed_size=config.embed_size,
                  hidden_size=config.hidden_size, dropout=config.dropout)
    inputs = [
        '我现在好得多了。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    for id, q in enumerate(inputs):
        print(q)
        print(m.predict(q))
        print()
# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import json
import os

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.evaluate import gen_target
from pycorrector.seq2seq_attention.seq2seq_attn_model import Seq2seqAttnModel


class Inference(object):
    def __init__(self, vocab_json_path='', attn_model_path='', maxlen=400):
        if os.path.exists(vocab_json_path):
            self.chars, id2char, self.char2id = json.load(open(vocab_json_path))
            self.id2char = {int(i): j for i, j in id2char.items()}
        else:
            print('not exist vocab path')
        seq2seq_attn_model = Seq2seqAttnModel(self.chars, attn_model_path=attn_model_path)
        self.model = seq2seq_attn_model.build_model()
        self.maxlen = maxlen

    def infer(self, sentence):
        return gen_target(sentence, self.model, self.char2id, self.id2char, self.maxlen, topk=3)


if __name__ == "__main__":
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    inference = Inference(vocab_json_path=config.vocab_json_path,
                          attn_model_path=config.attn_model_path,
                          maxlen=400)
    for i in inputs:
        target = inference.infer(i)
        print('input:' + i)
        print('output:' + target)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

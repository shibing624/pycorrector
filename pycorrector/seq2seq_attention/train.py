# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import json
import os

import numpy as np

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import CGEDReader, str2id, padding
from pycorrector.seq2seq_attention.evaluate import Evaluate
from pycorrector.seq2seq_attention.seq2seq_attn_model import Seq2seqAttnModel


def data_generator(input_texts, target_texts, char2id, batch_size, maxlen=400):
    # 数据生成器
    while True:
        X, Y = [], []
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i], char2id, maxlen))
            Y.append(str2id(target_texts[i], char2id, maxlen))
            if len(X) == batch_size:
                X = np.array(padding(X, char2id))
                Y = np.array(padding(Y, char2id))
                yield [X, Y], None
                X, Y = [], []


def train(train_path='', vocab_json_path='', attn_model_path='',
          batch_size=64, epochs=100, maxlen=400, hidden_dim=128):
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path)

    if os.path.exists(vocab_json_path):
        chars, id2char, char2id = json.load(open(vocab_json_path))
        id2char = {int(i): j for i, j in id2char.items()}
    else:
        print('Training data...')
        print('input_texts:', input_texts[0])
        print('target_texts:', target_texts[0])
        max_input_texts_len = max([len(text) for text in input_texts])

        print('num of samples:', len(input_texts))
        print('max sequence length for inputs:', max_input_texts_len)

        chars = data_reader.read_vocab(input_texts + target_texts)
        id2char = {i: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        json.dump([chars, id2char, char2id], open(vocab_json_path, 'w'))

    model = Seq2seqAttnModel(chars,
                             attn_model_path=attn_model_path,
                             hidden_dim=hidden_dim).build_model()
    evaluator = Evaluate(model, attn_model_path, char2id, id2char, maxlen)
    model.fit_generator(data_generator(input_texts, target_texts, char2id, batch_size, maxlen),
                        steps_per_epoch=(len(input_texts) + batch_size - 1) // batch_size,
                        epochs=epochs,
                        callbacks=[evaluator])


if __name__ == "__main__":
    train(train_path=config.train_path,
          vocab_json_path=config.vocab_json_path,
          attn_model_path=config.attn_model_path,
          batch_size=config.batch_size,
          epochs=config.epochs,
          maxlen=config.maxlen,
          hidden_dim=config.rnn_hidden_dim)

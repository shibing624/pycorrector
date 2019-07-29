# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
import os

import numpy as np

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import CGEDReader, str2id, padding, load_word_dict, save_word_dict
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


def get_validation_data(input_texts, target_texts, char2id, maxlen=400):
    # 数据生成器
    X, Y = [], []
    for i in range(len(input_texts)):
        X.append(str2id(input_texts[i], char2id, maxlen))
        Y.append(str2id(target_texts[i], char2id, maxlen))
        X = np.array(padding(X, char2id))
        Y = np.array(padding(Y, char2id))
        return [X, Y], None


def train(train_path='', test_path='', save_vocab_path='', attn_model_path='',
          batch_size=64, epochs=100, maxlen=400, hidden_dim=128, dropout=0.2, gpu_id=0):
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path)
    test_input_texts, test_target_texts = data_reader.build_dataset(test_path)

    # load or save word dict
    if os.path.exists(save_vocab_path):
        char2id = load_word_dict(save_vocab_path)
        id2char = {int(j): i for i, j in char2id.items()}
        chars = set([i for i in char2id.keys()])
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
        save_word_dict(char2id, save_vocab_path)

    model = Seq2seqAttnModel(len(char2id),
                             attn_model_path=attn_model_path,
                             hidden_dim=hidden_dim,
                             dropout=dropout,
                             gpu_id=gpu_id
                             ).build_model()
    evaluator = Evaluate(model, attn_model_path, char2id, id2char, maxlen)
    model.fit_generator(data_generator(input_texts, target_texts, char2id, batch_size, maxlen),
                        steps_per_epoch=(len(input_texts) + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=get_validation_data(test_input_texts, test_target_texts, char2id, maxlen),
                        callbacks=[evaluator])


if __name__ == "__main__":
    train(train_path=config.train_path,
          test_path=config.test_path,
          save_vocab_path=config.save_vocab_path,
          attn_model_path=config.attn_model_path,
          batch_size=config.batch_size,
          epochs=config.epochs,
          maxlen=config.maxlen,
          hidden_dim=config.rnn_hidden_dim,
          dropout=config.dropout,
          gpu_id=config.gpu_id)

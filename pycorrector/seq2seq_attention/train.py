# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping

sys.path.append('../..')

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.data_reader import build_dataset, read_vocab, str2id, padding, load_word_dict, \
    save_word_dict, GO_TOKEN, EOS_TOKEN
from pycorrector.seq2seq_attention.evaluate import Evaluate
from pycorrector.seq2seq_attention.seq2seq_attn_model import Seq2seqAttnModel


def data_generator(input_texts, target_texts, vocab2id, batch_size, maxlen=400):
    # 数据生成器
    while True:
        X, Y = [], []
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i], vocab2id, maxlen))
            Y.append([vocab2id[GO_TOKEN]] + str2id(target_texts[i], vocab2id, maxlen) + [vocab2id[EOS_TOKEN]])
            if len(X) == batch_size:
                X = np.array(padding(X, vocab2id))
                Y = np.array(padding(Y, vocab2id))
                yield [X, Y], None
                X, Y = [], []


def get_validation_data(input_texts, target_texts, vocab2id, maxlen=400):
    # 数据生成器
    X, Y = [], []
    for i in range(len(input_texts)):
        X.append(str2id(input_texts[i], vocab2id, maxlen))
        Y.append([vocab2id[GO_TOKEN]] + str2id(target_texts[i], vocab2id, maxlen) + [vocab2id[EOS_TOKEN]])
        X = np.array(padding(X, vocab2id))
        Y = np.array(padding(Y, vocab2id))
        return [X, Y], None


def train(train_path='', test_path='', save_vocab_path='', attn_model_path='',
          batch_size=64, epochs=100, maxlen=400, hidden_dim=128, dropout=0.2,
          vocab_max_size=50000, vocab_min_count=5, gpu_id=0):
    source_texts, target_texts = build_dataset(train_path)
    test_input_texts, test_target_texts = build_dataset(test_path)

    # load or save word dict
    if os.path.exists(save_vocab_path):
        vocab2id = load_word_dict(save_vocab_path)
    else:
        print('Training data...')
        vocab2id = read_vocab(source_texts + target_texts, max_size=vocab_max_size, min_count=vocab_min_count)
        num_encoder_tokens = len(vocab2id)
        max_input_texts_len = max([len(text) for text in source_texts])

        print('input_texts:', source_texts[0])
        print('target_texts:', target_texts[0])
        print('num of samples:', len(source_texts))
        print('num of unique input tokens:', num_encoder_tokens)
        print('max sequence length for inputs:', max_input_texts_len)
        save_word_dict(vocab2id, save_vocab_path)

    id2vocab = {int(j): i for i, j in vocab2id.items()}
    print('The vocabulary file:%s, size: %s' % (save_vocab_path, len(vocab2id)))
    model = Seq2seqAttnModel(len(vocab2id),
                             attn_model_path=attn_model_path,
                             hidden_dim=hidden_dim,
                             dropout=dropout,
                             gpu_id=gpu_id
                             ).build_model()
    evaluator = Evaluate(model, attn_model_path, vocab2id, id2vocab, maxlen)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    model.fit_generator(data_generator(source_texts, target_texts, vocab2id, batch_size, maxlen),
                        steps_per_epoch=(len(source_texts) + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=get_validation_data(test_input_texts, test_target_texts, vocab2id, maxlen),
                        callbacks=[evaluator, earlystop])


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
          vocab_max_size=config.vocab_max_size,
          vocab_min_count=config.vocab_min_count,
          gpu_id=config.gpu_id)

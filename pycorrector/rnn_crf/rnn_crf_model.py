# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: BiLSTM-CRF

from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import Sequential
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


def create_model(word_dict, label_dict, embedding_dim=100, rnn_hidden_dim=200, dropout=0.5):
    # build model
    model = Sequential()
    # embedding
    model.add(Embedding(len(word_dict), embedding_dim, mask_zero=True))
    # bilstm
    model.add(Bidirectional(LSTM(rnn_hidden_dim // 2, return_sequences=True,
                                 recurrent_dropout=dropout)))
    # crf
    crf = CRF(len(label_dict), sparse_target=True)
    model.add(crf)
    # loss
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def save_model(model, save_model_path):
    save_load_utils.save_all_weights(model, save_model_path)


def load_model(word_dict, label_dict, embedding_dim, rnn_hidden_dim, dropout, save_model_path):
    # https://github.com/keras-team/keras-contrib/issues/125
    model = create_model(word_dict, label_dict, embedding_dim, rnn_hidden_dim, dropout)
    save_load_utils.load_all_weights(model, save_model_path, include_optimizer=False)
    return model

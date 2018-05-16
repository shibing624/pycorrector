# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: BiLSTM

from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model

from utils.io_utils import get_logger

logger = get_logger(__name__)


def create_model(word_dict, embedding_dim=100, rnn_hidden_dim=200, dropout=0.5, maxlen=300):
    logger.info('build bilstm language model')
    # build model
    model = Sequential()
    # embedding
    model.add(Embedding(len(word_dict), embedding_dim))
    # bilstm
    # model.add(LSTM(rnn_hidden_dim//2, return_sequences=True))
    # model.add(Dropout(dropout))
    # model.add(LSTM(rnn_hidden_dim//2, return_sequences=False))
    # model.add(Dropout(dropout))
    # bilstm
    model.add(Bidirectional(LSTM(rnn_hidden_dim // 2, return_sequences=True,
                                 recurrent_dropout=dropout)))
    model.add(Dense(len(word_dict), activation='softmax'))
    # loss
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model.summary())
    return model


def load_model(save_model_path):
    return load_model(save_model_path)

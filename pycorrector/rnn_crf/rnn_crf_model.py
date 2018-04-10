# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: BiLSTM-CRF

from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import Sequential
from keras_contrib.layers import CRF


def rnn_crf(X_train, y_train, X_test, y_test, batch_size, word_dict,
            label_dict, embedding_dim, rnn_hidden_dim, epoch):
    model = Sequential()
    # embedding
    model.add(Embedding(len(word_dict), embedding_dim, mask_zero=True))
    # bilstm
    model.add(Bidirectional(LSTM(rnn_hidden_dim // 2, return_sequences=True,
                                 recurrent_dropout=0.5)))
    # crf
    crf = CRF(len(label_dict), sparse_target=True)
    model.add(crf)
    # loss
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
              validation_data=[X_test, y_test])
    return model

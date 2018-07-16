# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: BiLSTM-CRF

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import LSTM
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
    model.summary()
    return model


def save_model(model, save_model_path):
    save_load_utils.save_all_weights(model, save_model_path)


def load_model(word_dict, label_dict, embedding_dim, rnn_hidden_dim, dropout, save_model_path):
    # https://github.com/keras-team/keras-contrib/issues/125
    model = create_model(word_dict, label_dict, embedding_dim, rnn_hidden_dim, dropout)
    # save_load_utils.load_all_weights(model, save_model_path, include_optimizer=False) # one way
    model.load_weights(save_model_path)  # another way
    return model


def callback(save_model_path, logger=None):
    # Print the batch number at the beginning of every batch.
    if logger:
        batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: logger.info('batch: %d' % batch))
    else:
        batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: print(batch))
    # define the checkpoint, save model
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_acc',
                                 save_best_only=True, mode='auto',
                                 save_weights_only=True, verbose=1)
    return [batch_print_callback, checkpoint]

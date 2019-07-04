# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import numpy as np

from pycorrector.rnn_crf.data_reader import build_dict
from pycorrector.rnn_crf.data_reader import load_dict
from pycorrector.rnn_crf.data_reader import pad_sequence
from pycorrector.rnn_crf.data_reader import vectorize_data
from pycorrector.rnn_crf.rnn_crf_model import callback
from pycorrector.rnn_crf.rnn_crf_model import create_model
from pycorrector.utils.io_utils import get_logger
from pycorrector.rnn_crf import config

logger = get_logger(__name__)
PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'


def train(train_word_path=None,
          train_label_path=None,
          word_dict_path=None,
          label_dict_path=None,
          save_model_path=None,
          batch_size=64,
          dropout=0.5,
          epoch=10,
          embedding_dim=100,
          rnn_hidden_dim=200,
          maxlen=300,
          cutoff_frequency=0):
    """
    Train the bilstm_crf model for grammar correction.
    """
    # build the word dictionary
    build_dict(train_word_path,
               word_dict_path,
               cutoff_frequency,
               insert_extra_words=[UNK_TOKEN, PAD_TOKEN])
    # build the label dictionary
    build_dict(train_label_path, label_dict_path)
    # load dict
    word_ids_dict = load_dict(word_dict_path)
    label_ids_dict = load_dict(label_dict_path)
    # read data to index
    word_ids = vectorize_data(train_word_path, word_ids_dict)
    label_ids = vectorize_data(train_label_path, label_ids_dict)
    max_len = np.max([len(i) for i in word_ids])
    print('max_len:', max_len)
    # pad sequence
    word_seq = pad_sequence(word_ids, maxlen=maxlen)
    label_seq = pad_sequence(label_ids, maxlen=maxlen)
    # reshape label for crf model use
    label_seq = np.reshape(label_seq, (label_seq.shape[0], label_seq.shape[1], 1))
    print(word_seq.shape)
    print(label_seq.shape)
    logger.info("Data loaded.")
    # model
    logger.info("Training BILSTM_CRF model...")
    model = create_model(word_ids_dict, label_ids_dict,
                         embedding_dim, rnn_hidden_dim, dropout)
    # callback
    callbacks_list = callback(save_model_path, logger)
    # fit
    model.fit(word_seq,
              label_seq,
              batch_size=batch_size,
              epochs=epoch,
              validation_split=0.2,
              callbacks=callbacks_list)
    logger.info("Training has finished.")


if __name__ == "__main__":
    train(train_word_path=config.train_word_path,
          train_label_path=config.train_label_path,
          word_dict_path=config.word_dict_path,
          label_dict_path=config.label_dict_path,
          save_model_path=config.save_model_path,
          batch_size=config.batch_size,
          dropout=config.dropout,
          epoch=config.epoch,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim,
          maxlen=config.maxlen,
          cutoff_frequency=config.cutoff_frequency)

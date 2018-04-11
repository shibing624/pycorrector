# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import numpy as np
from sklearn.model_selection import train_test_split

import rnn_crf_config as config
from data_reader import build_dict
from data_reader import get_max_len
from data_reader import load_dict
from data_reader import load_reverse_dict
from data_reader import pad_sequence
from data_reader import vectorize_data
from rnn_crf_model import create_model
from rnn_crf_model import save_model
from utils.io_utils import get_logger

logger = get_logger(__name__)
PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'


def train(train_word_path=None,
          train_label_path=None,
          word_dict_path=None,
          label_dict_path=None,
          save_model_path=None,
          batch_size=64,
          epoch=10,
          embedding_dim=100,
          rnn_hidden_dim=200,
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
    # max length of each sequence
    word_maxlen = get_max_len(word_ids)
    label_maxlen = get_max_len(label_ids)
    # pad sequence
    word_seq, label_seq = pad_sequence(word_ids, label_ids, word_maxlen, label_maxlen)
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        word_seq, label_seq, test_size=0.2, random_state=42)
    # reshape for crf model use
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)
    logger.info("Data loaded.")

    logger.info("Training BILSTM_CRF model...")
    model = create_model(word_ids_dict, label_ids_dict,
                         embedding_dim, rnn_hidden_dim)
    # fit
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=[X_test, y_test])
    # save model
    save_model(model, save_model_path)
    logger.info("model saved: %s" % save_model_path)
    logger.info("Training has finished.")


if __name__ == "__main__":
    train(train_word_path=config.train_word_path,
          train_label_path=config.train_label_path,
          word_dict_path=config.word_dict_path,
          label_dict_path=config.label_dict_path,
          save_model_path=config.save_model_path,
          batch_size=config.batch_size,
          epoch=config.epoch,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim,
          cutoff_frequency=config.cutoff_frequency)

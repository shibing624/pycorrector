# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from sklearn.model_selection import train_test_split
import numpy as np
import rnn_crf_config as config
from data_reader import build_dict
from data_reader import data_reader
from data_reader import get_max_len
from data_reader import load_dict
from data_reader import load_reverse_dict
from data_reader import pad_sequence
from rnn_crf_model import rnn_crf
from utils.io_utils import get_logger

logger = get_logger(__name__)
PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'


def train(word_data_path=None,
          label_data_path=None,
          word_dict_path=None,
          label_dict_path=None,
          batch_size=64,
          epoch=10,
          embedding_dim=100,
          rnn_hidden_dim=200,
          cutoff_frequency=0):
    """
    Train the bilstm_crf model for grammar correction.
    """
    # build the word dictionary
    build_dict(word_data_path,
               word_dict_path,
               cutoff_frequency,
               insert_extra_words=[UNK_TOKEN, PAD_TOKEN])
    # build the label dictionary
    build_dict(label_data_path, label_dict_path)
    # load dict
    word_ids_dict, ids_word_dict = load_dict(word_dict_path), load_reverse_dict(word_dict_path)
    label_ids_dict, ids_label_dict = load_dict(label_dict_path), load_reverse_dict(label_dict_path)
    # Read data to index
    word_ids = data_reader(word_data_path, word_ids_dict)
    label_ids = data_reader(label_data_path, label_ids_dict)
    maxlen = get_max_len(word_ids)
    # pad sequence
    word_seq, label_seq = pad_sequence(word_ids, label_ids, maxlen=maxlen)
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        word_seq, label_seq, test_size=0.2, random_state=42)
    # reshape for crf
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)
    model = rnn_crf(X_train, y_train, X_test, y_test, batch_size, word_ids_dict, label_ids_dict,
                    embedding_dim, rnn_hidden_dim, epoch)
    # predict
    y_test_pred = model.predict(word_seq).argmax(-1)
    # save predict file


    logger.info("Training has finished.")


if __name__ == "__main__":
    train(word_data_path=config.word_data_path,
          label_data_path=config.label_data_path,
          word_dict_path=config.word_dict_path,
          label_dict_path=config.label_dict_path,
          batch_size=config.batch_size,
          epoch=config.epoch,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim,
          cutoff_frequency=config.cutoff_frequency)

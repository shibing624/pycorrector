# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append('.')

import rnn_lm_config as config
from rnn_lm.data_reader import build_dict
from rnn_lm.data_reader import load_dict
from rnn_lm.data_reader import pad_sequence
from rnn_lm.data_reader import vectorize_data
from rnn_lm_model import create_model
from utils.io_utils import get_logger

logger = get_logger(__name__)


def train(train_word_path=None,
          word_dict_path=None,
          save_model_path=None,
          batch_size=64,
          dropout=0.2,
          epoch=10,
          embedding_dim=100,
          rnn_hidden_dim=200,
          maxlen=300,
          cutoff_frequency=0):
    """
    Train the bilstm lm model for grammar correction.
    """
    # build the word dictionary
    build_dict(train_word_path,
               word_dict_path,
               cutoff_frequency)
    # load dict
    word_ids_dict = load_dict(word_dict_path)
    # read data to index
    word_ids = vectorize_data(train_word_path, word_ids_dict)
    # pad sequence
    word_seq = pad_sequence(word_ids, maxlen=maxlen)
    logger.info("Data loaded.")
    # model
    logger.info("Training BILSTM LM model...")
    model = create_model(word_ids_dict,
                         embedding_dim=embedding_dim,
                         rnn_hidden_dim=rnn_hidden_dim,
                         dropout=dropout)
    # fit
    model.fit(word_seq,word_seq,
              batch_size=batch_size,
              epochs=epoch)
    # save model
    model.save(save_model_path)
    logger.info("Training has finished.")


if __name__ == "__main__":
    train(train_word_path=config.train_word_path,
          word_dict_path=config.word_dict_path,
          save_model_path=config.save_model_path,
          batch_size=config.batch_size,
          dropout=config.dropout,
          epoch=config.epoch,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim,
          maxlen=config.maxlen,
          cutoff_frequency=config.cutoff_frequency)

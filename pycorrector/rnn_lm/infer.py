# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import rnn_crf_config as config
from rnn_lm.data_reader import load_dict
from rnn_lm.data_reader import load_reverse_dict
from rnn_lm.data_reader import load_test_id
from rnn_lm.data_reader import pad_sequence
from rnn_lm.data_reader import vectorize_data
from rnn_lm_model import load_model
from utils.io_utils import get_logger

logger = get_logger(__name__)


def infer(save_model_path, test_id_path, test_word_path,
          word_dict_path=None,  batch_size=64, maxlen=300):
    # load dict
    test_ids = load_test_id(test_id_path)
    word_ids_dict, ids_word_dict = load_dict(word_dict_path), load_reverse_dict(word_dict_path)
    # read data to index
    word_ids = vectorize_data(test_word_path, word_ids_dict)
    # pad sequence
    word_seq = pad_sequence(word_ids, maxlen)
    # load model by file
    model = load_model(save_model_path)
    probs = model.predict(word_seq, batch_size=batch_size).argmax(-1)
    assert len(probs) == len(word_seq)
    print('probs.shape:', probs.shape)


if __name__ == '__main__':
    infer(config.save_model_path,
          config.test_id_path,
          config.test_word_path,
          config.test_label_path,
          word_dict_path=config.word_dict_path,
          label_dict_path=config.label_dict_path,
          save_pred_path=config.save_pred_path,
          batch_size=config.batch_size,
          dropout=config.dropout,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim,
          maxlen=config.maxlen)

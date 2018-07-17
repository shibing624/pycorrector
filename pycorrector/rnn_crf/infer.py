# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from pycorrector.rnn_crf import rnn_crf_config as config
from pycorrector.rnn_crf.data_reader import load_dict
from pycorrector.rnn_crf.data_reader import load_reverse_dict
from pycorrector.rnn_crf.data_reader import load_test_id
from pycorrector.rnn_crf.data_reader import pad_sequence
from pycorrector.rnn_crf.data_reader import vectorize_data
from pycorrector.rnn_crf.rnn_crf_model import load_model
from pycorrector.utils.io_utils import get_logger

logger = get_logger(__name__)


def infer(save_model_path, test_id_path, test_word_path, test_label_path,
          word_dict_path=None, label_dict_path=None, save_pred_path=None,
          batch_size=64, dropout=0.5, embedding_dim=100,
          rnn_hidden_dim=200, maxlen=300):
    # load dict
    test_ids = load_test_id(test_id_path)
    word_ids_dict, ids_word_dict = load_dict(word_dict_path), load_reverse_dict(word_dict_path)
    label_ids_dict, ids_label_dict = load_dict(label_dict_path), load_reverse_dict(label_dict_path)
    # read data to index
    word_ids = vectorize_data(test_word_path, word_ids_dict)
    # pad sequence
    word_seq = pad_sequence(word_ids, maxlen)
    # load model by file
    model = load_model(word_ids_dict, label_ids_dict, embedding_dim,
                       rnn_hidden_dim, dropout, save_model_path)
    probs = model.predict(word_seq, batch_size=batch_size, verbose=0).argmax(-1)
    assert len(probs) == len(word_seq)
    print('probs.shape:', probs.shape)

    test_words, test_labels = load_test_id(test_word_path), load_test_id(test_label_path)
    save_preds(probs, test_ids, word_seq, ids_word_dict,
               label_ids_dict, ids_label_dict, save_pred_path, test_words, test_labels)


def save_preds(preds, test_ids, X_test, ids_word_dict,
               label_ids_dict, ids_label_dict, out_path, test_words, test_labels):
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(len(X_test)):
            sent_ids = X_test[i]
            sid = test_ids[i]
            sentence = test_words[i]
            gold_error = test_labels[i]
            label = []
            for j in range(len(sent_ids)):
                if sent_ids[j] != 0:
                    label.append(preds[i][j])
            print(label)
            continue_error = False
            has_error = False
            current_error = 0
            start_pos = 0
            for k in range(len(label)):
                error_label_id = is_error_label_id(label[k], label_ids_dict)
                if error_label_id and not continue_error:
                    continue_error = True
                    start_pos = k + 1
                    current_error = label[k]
                    has_error = True
                if continue_error and label[k] != current_error and not error_label_id:
                    end_pos = k
                    f.write('%s\t%d\t%d\t%s\t%s\t%s\n' % (sid, start_pos, end_pos,
                                                          ids_label_dict[current_error], sentence, gold_error))
                    continue_error = False
                    current_error = 0
                if continue_error and label[k] != current_error and error_label_id:
                    end_pos = k
                    f.write('%s\t%d\t%d\t%s\t%s\t%s\n' % (sid, start_pos, end_pos,
                                                          ids_label_dict[current_error], sentence, gold_error))
                    start_pos = k + 1
                    current_error = label[k]
            if not has_error:
                f.write('%s\tcorrect\t%s\t%s\n' % (sid, sentence, gold_error))
        logger.info('save to %s done, data size: %d' % (out_path, len(X_test)))


def is_error_label_id(label_id, label_ids_dict):
    # return label_id != label_ids_dict['O']
    return label_id == label_ids_dict['M'] or label_id == label_ids_dict['R'] or label_id == label_ids_dict[
        'S'] or label_id == label_ids_dict['W']


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

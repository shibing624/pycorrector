# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import rnn_crf_config as config
from data_reader import get_max_len
from data_reader import load_dict
from data_reader import load_reverse_dict
from data_reader import load_test_id
from data_reader import pad_sequence
from data_reader import vectorize_data
from rnn_crf_model import load_model


def infer(save_model_path, test_id_path, test_word_path, test_label_path,
          word_dict_path=None, label_dict_path=None, save_pred_path=None,
          batch_size=64, embedding_dim=100, rnn_hidden_dim=200):
    # load dict
    test_ids = load_test_id(test_id_path)
    word_ids_dict, ids_word_dict = load_dict(word_dict_path), load_reverse_dict(word_dict_path)
    label_ids_dict, ids_label_dict = load_dict(label_dict_path), load_reverse_dict(label_dict_path)
    # read data to index
    word_ids = vectorize_data(test_word_path, word_ids_dict)
    label_ids = vectorize_data(test_label_path, label_ids_dict)
    # max length of each sequence
    word_maxlen = get_max_len(word_ids)
    label_maxlen = get_max_len(label_ids)
    # pad sequence
    word_seq, label_seq = pad_sequence(word_ids, label_ids, word_maxlen, label_maxlen)
    # load model by file
    model = load_model(word_ids_dict, label_ids_dict, embedding_dim, rnn_hidden_dim, save_model_path)
    probs = model.predict(word_seq, batch_size=batch_size).argmax(-1)
    assert len(probs) == len(label_seq)
    print('probs.shape:', probs.shape)
    save_preds(probs, test_ids, ids_word_dict, label_ids_dict, ids_label_dict, word_seq, save_pred_path)


def save_preds(preds, test_ids , ids_word_dict, label_ids_dict, ids_label_dict, X_test, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(len(X_test)):
            sent = X_test[i]
            sid = test_ids[i]
            sentence = ''.join([ids_word_dict[i] for i in sent if i > 0])
            label = []
            for j in range(len(sent)):
                if sent[j] != 0:
                    label.append(preds[i][j])
            error_flag = False
            is_correct = False
            current_error = 0
            start_pos = 0
            for k in range(len(label)):
                if (label[k] == label_ids_dict['R'] or label[k] == label_ids_dict['S'] or \
                                label[k] == label_ids_dict['M'] or \
                                label[k] == label_ids_dict['W']) and error_flag == False:
                    error_flag = True
                    start_pos = k + 1
                    current_error = label[k]
                    is_correct = True

                if error_flag and label[k] != current_error and (
                                        label[k] != label_ids_dict['R'] and label[k] != label_ids_dict['S'] and \
                                        label[k] != label_ids_dict['M'] and label[k] != label_ids_dict['W']):
                    end_pos = k
                    f.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, ids_label_dict[current_error]))

                    error_flag = False
                    current_error = 0

                if error_flag and label[k] != current_error and (
                                        label[k] == label_ids_dict['R'] or label[k] == label_ids_dict['S'] or \
                                        label[k] == label_ids_dict['M'] or label[k] == label_ids_dict['W']):
                    end_pos = k
                    f.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, ids_label_dict[current_error]))

                    start_pos = k + 1
                    current_error = label[k]
            if not is_correct:
                f.write('%s, correct\n' % (sid))
        print('done, infer data size: %d' % len(X_test))


if __name__ == '__main__':
    infer(config.save_model_path,
          config.test_id_path,
          config.test_word_path,
          config.test_label_path,
          word_dict_path=config.word_dict_path,
          label_dict_path=config.label_dict_path,
          save_pred_path=config.save_pred_path,
          batch_size=config.batch_size,
          embedding_dim=config.embedding_dim,
          rnn_hidden_dim=config.rnn_hidden_dim)

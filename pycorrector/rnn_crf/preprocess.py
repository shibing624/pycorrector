# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from xml.dom import minidom

import rnn_crf_config as config
from utils.text_utils import segment


def load_corpus_data(data_path):
    word_lst, label_lst = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        dom_tree = minidom.parse(f)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        sentence = doc.getElementsByTagName('TEXT')[0]. \
            childNodes[0].data.strip()
        errors = doc.getElementsByTagName('ERROR')
        # Find the error position and error type
        locate_dict = {}
        for error in errors:
            start_off = error.getAttribute('start_off')
            end_off = error.getAttribute('end_off')
            error_type = error.getAttribute('type')
            for i in range(int(start_off) - 1, int(end_off)):
                locate_dict[i] = error_type
        # Segment with pos
        word_seq, pos_seq = segment(sentence, cut_type='char', pos=True)
        word_arr, label_arr = [], []
        for i in range(len(word_seq)):
            if i in locate_dict:
                word_arr.append(word_seq[i])
                # Fill with error type
                label_arr.append(locate_dict[i])
            else:
                word_arr.append(word_seq[i])
                # Fill with pos tag
                label_arr.append(pos_seq[i])
        word_lst.append(word_arr)
        label_lst.append(label_arr)

    return word_lst, label_lst


def transform_corpus_data(data_list, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data_list:
            f.write(' '.join(line) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


if __name__ == '__main__':
    # train data
    train_words, train_labels = [], []
    for path in config.train_paths:
        word_list, label_list = load_corpus_data(path)
        train_words.extend(word_list)
        train_labels.extend(label_list)
    transform_corpus_data(train_words, config.train_word_path)
    transform_corpus_data(train_labels, config.train_label_path)

    # test data
    test_words, test_labels = [], []
    for path in config.test_paths:
        word_list, label_list = load_corpus_data(path)
        test_words.extend(word_list)
        test_labels.extend(label_list)
    transform_corpus_data(test_words, config.test_word_path)
    transform_corpus_data(test_labels, config.test_label_path)
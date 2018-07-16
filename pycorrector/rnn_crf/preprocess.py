# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from xml.dom import minidom

import pycorrector.rnn_crf.rnn_crf_config as config
from pycorrector.utils.text_utils import segment


def parse_xml_file(path):
    print('Parse data from %s' % path)
    id_lst, word_lst, label_lst = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        dom_tree = minidom.parse(path)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('TEXT')[0]. \
            childNodes[0].data.strip()
        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')
        errors = doc.getElementsByTagName('ERROR')
        # Locate the error position and error type
        locate_dict = {}
        for error in errors:
            start_off = error.getAttribute('start_off')
            end_off = error.getAttribute('end_off')
            error_type = error.getAttribute('type')
            for i in range(int(start_off) - 1, int(end_off)):
                if i == int(start_off) - 1:
                    error_type_change = 'B-' + error_type
                else:
                    error_type_change = 'I-' + error_type
                # locate_dict[i] = error_type_change
                locate_dict[i] = error_type
        # Segment with pos
        word_seq, pos_seq = segment(text, cut_type='char', pos=True)
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
        id_lst.append(text_id)
        word_lst.append(word_arr)
        label_lst.append(label_arr)
    return id_lst, word_lst, label_lst


def parse_txt_file(input_path, truth_path):
    print('Parse data from %s and %s' % (input_path, truth_path))
    id_lst, word_lst, label_lst = [], [], []
    # read truth file
    truth_dict = {}
    with open(truth_path, 'r', encoding='utf-8') as truth_f:
        for line in truth_f:
            parts = line.strip().split(',')
            # Locate the error position
            locate_dict = {}
            if len(parts) == 4:
                text_id = parts[0]
                start_off = parts[1]
                end_off = parts[2]
                error_type = parts[3].strip()
                for i in range(int(start_off) - 1, int(end_off)):
                    if i == int(start_off) - 1:
                        error_type_change = 'B-' + error_type
                    else:
                        error_type_change = 'I-' + error_type
                    # locate_dict[i] = error_type_change
                    locate_dict[i] = error_type
                # for i in range(int(start_off) - 1, int(end_off)):
                #     locate_dict[i] = error_type
                if text_id in truth_dict:
                    truth_dict[text_id].update(locate_dict)
                else:
                    truth_dict[text_id] = locate_dict

    # read input file and get tokenize
    with open(input_path, 'r', encoding='utf-8') as input_f:
        for line in input_f:
            parts = line.strip().split('\t')
            text_id = parts[0].replace('(sid=', '').replace(')', '')
            text = parts[1]
            # segment with pos
            word_seq, pos_seq = segment(text, cut_type='char', pos=True)
            word_arr, label_arr = [], []
            if text_id in truth_dict:
                locate_dict = truth_dict[text_id]
                for i in range(len(word_seq)):
                    if i in locate_dict:
                        word_arr.append(word_seq[i])
                        # fill with error type
                        label_arr.append(locate_dict[i])
                    else:
                        word_arr.append(word_seq[i])
                        # fill with pos tag
                        label_arr.append(pos_seq[i])
            else:
                word_arr = word_seq
                label_arr = pos_seq
            id_lst.append(text_id)
            word_lst.append(word_arr)
            label_lst.append(label_arr)
    return id_lst, word_lst, label_lst


def save_data_list(data_list, data_path):
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
        _, word_list, label_list = parse_xml_file(path)
        train_words.extend(word_list)
        train_labels.extend(label_list)
    save_data_list(train_words, config.train_word_path)
    save_data_list(train_labels, config.train_label_path)

    # test data
    test_ids, test_words, test_labels = [], [], []
    for input_path, truth_path in config.test_paths.items():
        id_list, word_list, label_list = parse_txt_file(input_path, truth_path)
        test_ids.extend(id_list)
        test_words.extend(word_list)
        test_labels.extend(label_list)
    save_data_list(test_ids, config.test_id_path)
    save_data_list(test_words, config.test_word_path)
    save_data_list(test_labels, config.test_label_path)

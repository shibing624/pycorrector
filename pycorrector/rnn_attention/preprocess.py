# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from codecs import open
from xml.dom import minidom

from sklearn.model_selection import train_test_split

import pycorrector.rnn_attention.config as config
from pycorrector.tokenizer import segment


def parse_xml_file(path):
    print('Parse data from %s' % path)
    data_list = []
    dom_tree = minidom.parse(path)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('TEXT')[0]. \
            childNodes[0].data.strip()
        # Input the correct text
        correction = doc.getElementsByTagName('CORRECTION')[0]. \
            childNodes[0].data.strip()
        # Segment
        source = segment(text, cut_type='char')
        target = segment(correction, cut_type='char')
        data_list.append([source, target])
    return data_list


def _save_data(data_list, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write('src: ' + ' '.join(src) + '\n')
            f.write('dst: ' + ' '.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def _save_data_part(data_list, data_path):
    data_path = data_path[:-4]
    with open(data_path + '.x.txt', 'w', encoding='utf-8') as f1, \
            open(data_path + '.y.txt', 'w', encoding='utf-8') as f2:
        count = 0
        for src, dst in data_list:
            f1.write(''.join(src) + '\n')
            f2.write(''.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def transform_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)


def transform_corpus_data_part(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list)
    _save_data_part(train_lst, train_data_path)
    _save_data_part(test_lst, test_data_path)


if __name__ == '__main__':
    # train data
    data_list = []
    for path in config.raw_train_paths:
        data_list.extend(parse_xml_file(path))
    # transform_corpus_data(data_list, config.train_path, config.test_path)
    transform_corpus_data_part(data_list, config.train_path, config.test_path)

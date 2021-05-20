# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import os
import sys

sys.path.append('../..')
from xml.dom import minidom

from pycorrector.deep_context import config
from pycorrector.utils.tokenizer import segment


def parse_xml_file(path):
    print('Parse data from %s' % path)
    word_arr = []
    with open(path, 'r', encoding='utf-8') as f:
        dom_tree = minidom.parse(f)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('CORRECTION')[0]. \
            childNodes[0].data.strip()
        # Segment
        word_seq = segment(text, cut_type='char', pos=False)
        word_arr.append(word_seq)
    return word_arr


def save_data_list(data_list, data_path):
    dirname = os.path.dirname(data_path)
    os.makedirs(dirname, exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data_list:
            f.write(' '.join(line) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


if __name__ == '__main__':
    # train data
    train_words = []
    for path in config.raw_train_paths:
        train_words.extend(parse_xml_file(path))
    save_data_list(train_words, config.train_path)

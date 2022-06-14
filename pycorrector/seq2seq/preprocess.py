# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import sys
from xml.dom import minidom

from sklearn.model_selection import train_test_split

sys.path.append('../..')
from pycorrector.utils.tokenizer import segment


def parse_xml_file(path, use_segment, segment_type):
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

        source = ' '.join(segment(text.strip(), cut_type=segment_type)) if use_segment else text.strip()
        target = ' '.join(segment(correction.strip(), cut_type=segment_type)) if use_segment else correction.strip()

        pair = [source, target]
        if pair not in data_list:
            data_list.append(pair)
    return data_list


def get_data_file(path, use_segment, segment_type):
    data_list = []
    if not os.path.exists(path):
        print('%s not exists' % path)
        return data_list
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            source = ' '.join(segment(parts[0].strip(), cut_type=segment_type)) if use_segment else parts[0].strip()
            target = ' '.join(segment(parts[1].strip(), cut_type=segment_type)) if use_segment else parts[1].strip()

            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, data_path):
    dirname = os.path.dirname(data_path)
    os.makedirs(dirname, exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write(src + '\t' + dst + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def save_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)

# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import os
import sys

from xml.dom import minidom

sys.path.append('../..')
from pycorrector.utils.tokenizer import segment
from pycorrector.deepcontext import config


def parse_xml_file(path, use_segment, segment_type):
    print('Parse data from %s' % path)
    word_arr = []
    dom_tree = minidom.parse(path)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('CORRECTION')[0]. \
            childNodes[0].data.strip()
        # Segment
        word_seq = ' '.join(segment(text.strip(), cut_type=segment_type)) if use_segment else text.strip()
        word_arr.append(word_seq)
    return word_arr


def get_data_file(path, use_segment, segment_type):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            target = ' '.join(segment(parts[1].strip(), cut_type=segment_type)) if use_segment else parts[1].strip()
            data_list.append(target)
    return data_list


def save_corpus_data(data_list, data_path):
    dirname = os.path.dirname(data_path)
    os.makedirs(dirname, exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data_list:
            f.write(line + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


if __name__ == '__main__':
    # train data
    data_list = []
    if config.dataset == 'sighan':
        data = get_data_file(config.sighan_train_path, config.use_segment, config.segment_type)
        data_list.extend(data)
    else:
        for path in config.cged_train_paths:
            data_list.extend(parse_xml_file(path, config.use_segment, config.segment_type))

    save_corpus_data(data_list, config.train_path)

# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import sys

sys.path.append('../../..')
from codecs import open
from xml.dom import minidom

from sklearn.model_selection import train_test_split

from pycorrector.transformer.tf import config
from pycorrector.utils.tokenizer import segment

split_symbol = ['，', '。', '？', '！']


def split_2_short_text(sentence):
    for i in split_symbol:
        sentence = sentence.replace(i, i + '\t')
    return sentence.split('\t')


def parse_xml_file(path, use_short_text=False, maximum_length=200):
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

        if use_short_text:
            texts = split_2_short_text(text)
            corrections = split_2_short_text(correction)
        else:
            texts = [text]
            corrections = [correction]
        if len(texts) != len(corrections):
            print('error diff:' + text + '\t' + correction)
            continue
        for i in range(len(texts)):
            if len(texts[i]) > maximum_length:
                print('error long:' + texts[i] + '\t' + corrections[i])
                continue
            source = segment(texts[i], cut_type='char')
            target = segment(corrections[i], cut_type='char')
            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, src_path, tgt_path):
    with open(src_path, 'w', encoding='utf-8') as f1, open(tgt_path, 'w', encoding='utf-8') as f2:
        count = 0
        for src, dst in data_list:
            f1.write(' '.join(src) + '\n')
            f2.write(' '.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s and %s" % (count, src_path, tgt_path))


def transform_corpus_data(data_list, train_src_path, train_tgt_path, test_src_path, test_tgt_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1)
    _save_data(train_lst, train_src_path, train_tgt_path)
    _save_data(test_lst, test_src_path, test_tgt_path)


if __name__ == '__main__':
    # train data
    data_list = []
    for path in config.raw_train_paths:
        data_list.extend(
            parse_xml_file(path, use_short_text=config.use_short_text, maximum_length=config.maximum_length))
    transform_corpus_data(data_list,
                          config.src_train_path,
                          config.tgt_train_path,
                          config.src_test_path,
                          config.tgt_test_path)

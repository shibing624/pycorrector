# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import os
import sys
from codecs import open
from xml.dom import minidom

from sklearn.model_selection import train_test_split

sys.path.append('../..')

from pycorrector.utils.tokenizer import segment
from pycorrector.transformer import config


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

        source = segment(text.strip(), cut_type='char')
        target = segment(correction.strip(), cut_type='char')

        pair = [source, target]
        if pair not in data_list:
            data_list.append(pair)
    return data_list


def save_data(data_list, src_data_path, trg_data_path):
    with open(src_data_path, 'w', encoding='utf-8') as f1, \
            open(trg_data_path, 'w', encoding='utf-8')as f2:
        count = 0
        for src, dst in data_list:
            f1.write(' '.join(src) + '\n')
            f2.write(' '.join(dst) + '\n')
            count += 1
        print("save line size:%d" % count)


def gen_fairseq_data(source_lang,
                     target_lang,
                     trainpref,
                     validpref,
                     nwordssrc,
                     nwordstgt,
                     destdir,
                     joined_dictionary
                     ):
    from fairseq import options
    from fairseq_cli import preprocess

    parser = options.get_preprocessing_parser()
    args = parser.parse_args()

    args.source_lang = source_lang
    args.target_lang = target_lang
    args.trainpref = trainpref
    args.validpref = validpref
    args.nwordssrc = nwordssrc
    args.nwordstgt = nwordstgt
    args.destdir = destdir
    args.joined_dictionary = joined_dictionary
    preprocess.main(args)


if __name__ == '__main__':
    # if exist download big data, only generate fairseq data
    if not os.path.exists(config.train_src_path):
        # not exist big data, generate toy train data
        data_list = []
        for path in config.raw_train_paths:
            data_list.extend(parse_xml_file(path))
        train_lst, val_lst = train_test_split(data_list, test_size=0.1)
        save_data(train_lst, config.train_src_path, config.train_trg_path)
        save_data(val_lst, config.val_src_path, config.val_trg_path)

    # generate fairseq format data with prepared train data
    gen_fairseq_data(config.train_src_path.split('.')[-1],
                     config.train_trg_path.split('.')[-1],
                     config.trainpref,
                     config.valpref,
                     config.vocab_max_size,
                     config.vocab_max_size,
                     config.data_bin_dir,
                     config.joined_dictionary
                     )

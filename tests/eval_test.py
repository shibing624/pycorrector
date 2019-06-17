# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append("../")
import os

from pycorrector.eval import *

pwd_path = os.path.abspath(os.path.dirname(__file__))
bcmi_path = os.path.join(pwd_path, '../pycorrector/data/cn/bcmi.txt')
clp_path = os.path.join(pwd_path, '../pycorrector/data/cn/clp14_C1.pkl')
sighan_path = os.path.join(pwd_path, '../pycorrector/data/cn/sighan15_A2.pkl')


def get_gcmi_cor_test():
    s = '青蛙是庄家的好朋友，我们要宝（（保））护它们。'
    print(get_bcmi_corpus(s))


def eval_bcmi_data_test():
    rate, right_dict, wrong_dict = eval_bcmi_data(bcmi_path, True)
    print('right rate:{}, right_dict:{}, wrong_dict:{}'.format(rate, right_dict, wrong_dict))


def clp_data_test():
    rate, right_dict, wrong_dict = eval_sighan_corpus(clp_path, True)
    print('right rate:{}, right_dict:{}, wrong_dict:{}'.format(rate, right_dict, wrong_dict))
    # cn_spell rate:0.2


def sighan_data_test():
    rate, right_dict, wrong_dict = eval_sighan_corpus(sighan_path, True)
    print('right rate:{}, right_dict:{}, wrong_dict:{}'.format(rate, right_dict, wrong_dict))
    # cn_spell rate:0.2


def get_confusion_dict():
    confusions = []
    sighan_data = load_pkl(clp_path)
    for error_sentence, right_detail in sighan_data:
        if right_detail:
            if right_detail[0][1:] not in confusions:
                confusions.append(right_detail[0][1:])

    sighan_data = load_pkl(sighan_path)
    for error_sentence, right_detail in sighan_data:
        if right_detail:
            if right_detail[0][1:] not in confusions:
                confusions.append(right_detail[0][1:])
    with open('a.txt', 'w', encoding='utf-8') as f:
        for i in confusions:
            f.write(i[0] + '\t' + i[1] + '\n')


if __name__ == "__main__":
    # get_gcmi_cor_test()
    # eval_bcmi_data_test()
    clp_data_test()
    sighan_data_test()
    # get_confusion_、dict()

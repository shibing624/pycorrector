# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append("../")
import os
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.eval import eval_bcmi_data, get_bcmi_corpus, eval_sighan_corpus

pwd_path = os.path.abspath(os.path.dirname(__file__))
bcmi_path = os.path.join(pwd_path, '../pycorrector/data/cn/bcmi.txt')
clp_path = os.path.join(pwd_path, '../pycorrector/data/cn/clp14_C1.pkl')
sighan_path = os.path.join(pwd_path, '../pycorrector/data/cn/sighan15_A2.pkl')


def get_bcmi_cor_test():
    s = '青蛙是庄家的好朋友，我们要宝（（保））护它们。'
    print(get_bcmi_corpus(s))


def eval_bcmi_data_test():
    rate, right_dict, wrong_dict = eval_bcmi_data(bcmi_path, True)
    print('right rate:{}, right_dict:{}, wrong_dict:{}'.format(rate, right_dict, wrong_dict))
    # right count: 104 ;sentence size: 383, right rate:0.271

def clp_data_test():
    rate = eval_sighan_corpus(clp_path, True)
    print('right rate:{}'.format(rate))
    # rate:1.6


def sighan_data_test():
    rate = eval_sighan_corpus(sighan_path, True)
    print('right rate:{}'.format(rate))
    # rate:1.53


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
    eval_bcmi_data_test()
    # clp_data_test()
    # sighan_data_test()
    # get_confusion_dict()

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append("../")
import os
from pycorrector.utils.eval import eval_bcmi_data, get_bcmi_corpus, eval_sighan_corpus

pwd_path = os.path.abspath(os.path.dirname(__file__))
bcmi_path = os.path.join(pwd_path, '../pycorrector/data/cn/bcmi.txt')
clp_path = os.path.join(pwd_path, '../pycorrector/data/cn/clp14_C1.pkl')
sighan_path = os.path.join(pwd_path, '../pycorrector/data/cn/sighan15_A2.pkl')
cged_path = os.path.join(pwd_path, '../pycorrector/data/cn/CGED/CGED16_HSK_TrainingSet.xml')


def test_get_bcmi_data():
    s = '青蛙是庄家的好朋友，我们要宝（（保））护它们。'
    print(get_bcmi_corpus(s))


def test_eval_bcmi_data():
    rate, right_dict, wrong_dict = eval_bcmi_data(bcmi_path, True)
    print('right rate:{}, right_dict:{}, wrong_dict:{}'.format(rate, right_dict, wrong_dict))
    # right count: 104 ;sentence size: 383, right rate:0.271


def test_clp_data():
    rate = eval_sighan_corpus(clp_path, True)
    print('right rate:{}'.format(rate))
    # rate:1.6


def test_sighan_data():
    rate = eval_sighan_corpus(sighan_path, True)
    print('right rate:{}'.format(rate))
    # rate:1.53

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os

from pycorrector.eval import *

pwd_path = os.path.abspath(os.path.dirname(__file__))
bcmi_path = os.path.join(pwd_path, '../pycorrector/data/cn/bcmi.txt')
clp_path = os.path.join(pwd_path, '../pycorrector/data/cn/clp14_C1.pkl')
sighan_path = os.path.join(pwd_path, '../pycorrector/data/cn/sighan15_A2.pkl')


def get_gcmi_cor_test():
    s = '老师工作非常幸（（辛））苦，我们要遵（（尊））敬老师。'
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


sighan_data_test()

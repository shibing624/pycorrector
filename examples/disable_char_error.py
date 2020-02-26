# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")

import pycorrector

if __name__ == '__main__':
    error_sentence_1 = '我的喉咙发炎了要买点阿莫细林吃'
    pycorrector.enable_char_error(enable=False)
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

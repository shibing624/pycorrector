# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import sys

sys.path.append("../")

import pycorrector


def test_text1():
    error_sentence_1 = '我的喉咙发炎了要买点阿莫细林吃'
    pycorrector.enable_char_error(enable=False)
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import sys

sys.path.append("../")

import pycorrector


def test_disease():
    """测试疾病名纠错"""
    pycorrector.enable_char_error(enable=False)
    error_sentence_1 = '这个新药奥美砂坦脂片能治疗心绞痛，效果还可以'  # 奥美沙坦酯片

    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '有个药名叫硫酸氢录吡各雷片能治疗高血压'  # 硫酸氢氯吡格雷片
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_brand():
    """测试品牌名纠错"""
    pycorrector.enable_char_error(enable=False)
    error_sentence_1 = '买衣服就到拼哆哆'  # 拼多多
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '这个特仑素牛奶喝起来还不错吧'  # 特仑苏
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_chengyu():
    """测试成语纠错"""
    pycorrector.enable_char_error(enable=False)
    error_sentence_1 = '这块名表带带相传'  # 代代相传
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '他贰话不说把牛奶喝完了'  # 二话不说
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_suyu():
    """测试俗语纠错"""
    pycorrector.enable_char_error(enable=False)

    error_sentence_1 = '这衣服买给她吧，也是肥水步流外人田'  # 肥水不流外人田
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '这么多字让他写也是赶鸭子打架'  # 赶鸭子上架
    correct_sent = pycorrector.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

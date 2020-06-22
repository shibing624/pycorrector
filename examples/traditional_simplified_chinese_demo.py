# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("..")

import pycorrector

if __name__ == '__main__':
    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = pycorrector.traditional2simplified(traditional_sentence)
    print(traditional_sentence, '=>', simplified_sentence)

    simplified_sentence = '忧郁的台湾乌龟'
    traditional_sentence = pycorrector.simplified2traditional(simplified_sentence)
    print(simplified_sentence, '=>', traditional_sentence)

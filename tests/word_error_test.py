# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
import pycorrector

text = [
    '我的宝贝万一值钱了呢',
    '我已经做了一遍工作',
]


def test1():
    for i in text:
        print(i, pycorrector.detect(i))
        print(i, pycorrector.correct(i))

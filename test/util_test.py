# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from util import *
from pypinyin import pinyin, lazy_pinyin

# traditional simplified
traditional_sentence = '憂郁的臺灣烏龜'
simplified_sentence = traditional2simplified(traditional_sentence)
print(simplified_sentence)

simplified_sentence = '忧郁的台湾乌龟'
traditional_sentence = simplified2traditional(simplified_sentence)
print(traditional_sentence)

print(pinyin('中心'))  # 带音调
print(pinyin('中心', heteronym=True))  # 多音字
print(lazy_pinyin('中心'))  # 不带音调

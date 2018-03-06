# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from pycorrector.util import *
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

print(preprocess('你干么！ｄ７＆８８８学英语ＡＢＣ？ｎｚ'))

print(tokenize('小姑娘蹦蹦跳跳的去了她外公家'))
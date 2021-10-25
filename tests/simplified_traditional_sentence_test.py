# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:

import sys

sys.path.append("../")
import opencc
# s2t.json Simplified Chinese to Traditional Chinese 簡體到繁體
# t2s.json Traditional Chinese to Simplified Chinese 繁體到簡體
s2t_converter = opencc.OpenCC('s2t.json')
t2s_converter = opencc.OpenCC('t2s.json')
print(s2t_converter.convert('汉字'))  # 漢字

traditional_sentence = '憂郁的臺灣烏龜'
simplified_sentence = t2s_converter.convert(traditional_sentence)
print(simplified_sentence)

simplified_sentence = '忧郁的台湾乌龟'
traditional_sentence = s2t_converter.convert(simplified_sentence)
print(traditional_sentence)


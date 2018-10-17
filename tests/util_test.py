# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from pypinyin import lazy_pinyin

from pycorrector.utils.text_utils import traditional2simplified, simplified2traditional
from pycorrector.utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin
from pycorrector.tokenizer import segment

traditional_sentence = '憂郁的臺灣烏龜'
simplified_sentence = traditional2simplified(traditional_sentence)
print(simplified_sentence)

simplified_sentence = '忧郁的台湾乌龟'
traditional_sentence = simplified2traditional(simplified_sentence)
print(traditional_sentence)

print(lazy_pinyin('中心'))  # 不带音调

print(segment('小姑娘蹦蹦跳跳的去了她外公家'))


pron = get_homophones_by_char('长')
print('get_homophones_by_char:', pron)

pron = get_homophones_by_pinyin('zha1ng')
print('get_homophones_by_pinyin:', pron)

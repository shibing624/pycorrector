# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
from pypinyin import lazy_pinyin
from pycorrector.utils.text_utils import traditional2simplified, simplified2traditional
from pycorrector.utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin
from pycorrector.utils.text_utils import is_chinese, is_chinese_string
from pycorrector.utils.tokenizer import split_text_by_maxlen, split_2_short_text


class UtilTestCase(unittest.TestCase):
    def test_single_util(self):
        traditional_sentence = '憂郁的臺灣烏龜'
        simplified_sentence = traditional2simplified(traditional_sentence)
        print(simplified_sentence)

        simplified_sentence = '忧郁的台湾乌龟'
        traditional_sentence = simplified2traditional(simplified_sentence)
        print(traditional_sentence)

        print(lazy_pinyin('中心'))  # 不带音调

        pron = get_homophones_by_char('长')
        print('get_homophones_by_char:', pron)

        pron = get_homophones_by_pinyin('zha1ng')
        print('get_homophones_by_pinyin:', pron)

        s = """现在 银色的K2P是MTK还是博通啊？李雯雯……“00后”选手
        啥123kMk.23?？ ''"’
        """
        print(s, is_chinese_string(s))

        for i in s:
            print(i, is_chinese(i))

    def test_split_maxlen(self):
        sents = [
            '智能手机中最好的是pingguo手机',
        ]
        res = []
        for name in sents:
            r = split_text_by_maxlen(name, 4)
            print(r)
            res.append(r)

        self.assertEqual(res[0][0], ('智能手机', 0))
        self.assertEqual(res[0][1], ('中最好的', 4))

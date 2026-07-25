# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
import unittest

sys.path.append('..')
from pycorrector import Corrector

m = Corrector()


class PinyinTestCase(unittest.TestCase):
    def test_single_pinyin(self):
        sents = [
            '我的宝贝万一zhuan钱了呢',
            '我已经zuo了一遍工作',
        ]
        res = []
        for name in sents:
            r = m.correct(name)['errors']
            print(r)
            res.append(r)

        # pinyin fragments are left untouched by the char/word corrector
        self.assertEqual(res[0], [])
        self.assertEqual(res[1], [])

    def test_full_pinyin(self):
        sents = [
            '你们要很xingfu才可以',
            '智能手机中最好的是pingguo手机',
        ]
        res = []
        for name in sents:
            r = m.correct(name)['errors']
            print(r)
            res.append(r)

        self.assertEqual(res[0], [])
        self.assertEqual(res[1], [])


if __name__ == '__main__':
    unittest.main()

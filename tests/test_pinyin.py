# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
import pycorrector


class PinyinTestCase(unittest.TestCase):
    def test_single_pinyin(self):
        sents = [
            '我的宝贝万一zhuan钱了呢',
            '我已经zuo了一遍工作',
        ]
        res = []
        for name in sents:
            s, r = pycorrector.correct(name)
            print(s, r)
            res.append(r)

        # self.assertEqual(res[0], [('zhuan', '赚', 6, 12)])
        # self.assertEqual(res[1], [('zuo', '做', 3, 7)])
        self.assertEqual(res[0], [])
        self.assertEqual(res[1], [])

    def test_full_pinyin(self):
        sents = [
            '你们要很xingfu才可以',
            '智能手机中最好的是pingguo手机',
        ]
        res = []
        for name in sents:
            s, r = pycorrector.correct(name)
            print(s, r)
            res.append(r)

        # self.assertEqual(res[0], [('xingfu', '幸福', 4, 11)])
        # self.assertEqual(res[1], [('pingguo', '苹果', 9, 17)])
        self.assertEqual(res[0], [])
        self.assertEqual(res[1], [])


if __name__ == '__main__':
    unittest.main()

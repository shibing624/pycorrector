# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
from pycorrector import MacBertCorrector

m = MacBertCorrector()


class MyTestCase(unittest.TestCase):
    def test1(self):
        sents = [
            '我们禅精竭虑学习',
            '禅精竭虑学习',
        ]
        res = []
        for i in sents:
            r = m.correct(i)
            print(i, r)
            res.append(r)

        self.assertEqual(res[0]['target'], '我们禅精竭虑学习')
        self.assertEqual(res[1]['target'], '禅精竭虑学习')


if __name__ == '__main__':
    unittest.main()

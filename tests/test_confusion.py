# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
import sys
import unittest

sys.path.append('..')

import pytest

pytest.importorskip("kenlm")

from pycorrector import Corrector

m = Corrector()
_CONFUSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'examples', 'kenlm', 'my_custom_confusion.txt')


class BaseTestCase(unittest.TestCase):
    def _assert_valid_result(self, corrected, source):
        self.assertIsInstance(corrected, dict)
        self.assertEqual(corrected['source'], source)
        self.assertIsInstance(corrected['target'], str)
        for wrong, right, pos in corrected['errors']:
            self.assertIsInstance(wrong, str)
            self.assertIsInstance(right, str)
            self.assertIsInstance(pos, int)

    def test_base_correct(self):
        query = '机七学习是人工智能领遇最能体现智能的一个分知'
        corrected = m.correct(query)
        print(corrected['target'], corrected['errors'])
        self._assert_valid_result(corrected, query)
        # the tiny LM consistently flags the first char-level confusion pair
        self.assertTrue(any(err[0] == '机七' for err in corrected['errors']))

    def test_base_demos(self):
        sents = [
            '少先队员因该为老人让坐',
            '今天心情很好',
            '真麻烦你了。希望你们好好的跳无',
            '机七学习是人工智能领遇最能体现智能的一个分知',
            '一只小鱼船浮在平净的河面上',
            '我的家乡是有明的渔米之乡',
        ]
        for name in sents:
            corrected = m.correct(name)
            print(corrected['errors'])
            self._assert_valid_result(corrected, name)

    def test_confusion_dict_file(self):
        sents = [
            '买iphonex，要多少钱',
            '共同实际控制人萧华、霍荣铨、张旗康',
            '通信用户份额呈现下降趋势',
        ]
        for name in sents:
            corrected = m.correct(name)
            print(corrected['errors'])
            self._assert_valid_result(corrected, name)

        m.set_custom_confusion_path_or_dict(_CONFUSION_FILE)
        corrected = m.correct(sents[0])
        print(corrected['errors'])
        self.assertIn(('iphonex', 'iphoneX', 1), corrected['errors'])

    def test_confusion_dict_dict(self):
        sents = [
            '买iphonex，要多少钱',
            '共同实际控制人萧华、霍荣铨、张旗康',
            '通信用户份额呈现下降趋势',
        ]
        for name in sents:
            print(name, m.detect(name))
            corrected = m.correct(name)
            print(corrected['errors'])
            self._assert_valid_result(corrected, name)
        print('*' * 42)

        m_dict = {'iphonex': 'iphoneX',
                  '张旗康': '张旗康',
                  '份额': '份额',
                  }

        m.set_custom_confusion_path_or_dict(m_dict)
        corrected = m.correct(sents[0])
        print(corrected['errors'])
        self.assertIn(('iphonex', 'iphoneX', 1), corrected['errors'])


if __name__ == '__main__':
    unittest.main()

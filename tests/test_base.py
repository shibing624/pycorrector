# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
import pycorrector


class BaseTestCase(unittest.TestCase):
    def test_base_correct(self):
        query = '机七学习是人工智能领遇最能体现智能的一个分知'
        corrected_sent, detail = pycorrector.correct(query)
        print(corrected_sent, detail)
        self.assertEqual(corrected_sent, '机器学习是人工智能领域最能体现智能的一个分知')
        self.assertEqual(detail, [('机七', '机器', 0, 2), ('领遇', '领域', 9, 11)])

    def test_base_demos(self):
        sents = [
            '少先队员因该为老人让坐',
            '今天心情很好',
            '真麻烦你了。希望你们好好的跳无',
            '机七学习是人工智能领遇最能体现智能的一个分知',
            '一只小鱼船浮在平净的河面上',
            '我的家乡是有明的渔米之乡',
        ]
        res = []
        for name in sents:
            s, r = pycorrector.correct(name)
            print(r)
            res.append(r)

        self.assertEqual(res[0], [('因该', '应该', 4, 6), ('坐', '座', 10, 11)])
        self.assertEqual(res[1], [])
        self.assertEqual(res[2], [('无', '舞', 14, 15)])
        self.assertEqual(res[3], [('机七', '机器', 0, 2), ('领遇', '领域', 9, 11)])
        self.assertEqual(res[4], [('平净', '平静', 7, 9)])
        self.assertEqual(res[5], [('有明', '有名', 5, 7)])

    def test_confusion_dict(self):
        sents = [
            '买iphonex，要多少钱',
            '共同实际控制人萧华、霍荣铨、张旗康',
        ]
        res = []
        for name in sents:
            s, r = pycorrector.correct(name)
            print(r)
            res.append(r)

        self.assertEqual(res[0], [])
        self.assertEqual(res[1], [('张旗康', '张启康', 14, 17)])

        pycorrector.set_custom_confusion_dict('../examples/my_custom_confusion.txt')
        res = []
        for name in sents:
            s, r = pycorrector.correct(name)
            print(r)
            res.append(r)
        self.assertEqual(res[0], [('iphonex', 'iphoneX', 1, 8)])
        self.assertEqual(res[1], [])


if __name__ == '__main__':
    unittest.main()

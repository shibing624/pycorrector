# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
import unittest

sys.path.append('..')
from pycorrector.en_spell import EnSpell
from pycorrector.utils.tokenizer import whitespace_tokenize


class EnTestCase(unittest.TestCase):
    def test_en_correct(self):
        """测试英文纠错"""
        en_correct = EnSpell().correct
        print(
            en_correct('spelling speling correctud gorrectud bycycle inconvient arrainged peotrry word quintessential'))
        print(en_correct('spelling')[0])
        assert en_correct('spelling')[0] == 'spelling'  # no error
        assert en_correct('speling')[0] == 'spelling'  # insert
        assert en_correct('correctud')[0] == 'corrected'  # replace 1
        assert en_correct('gorrectud')[0] == 'corrected'  # replace 2
        assert en_correct('bycycle')[0] == 'bicycle'  # replace
        assert en_correct('inconvient')[0] == 'inconvenient'  # insert 2
        assert en_correct('arrainged')[0] == 'arranged'  # delete
        assert en_correct('peotrry')[0] == 'poetry'  # transpose + delete
        assert en_correct('word')[0] == 'word'  # know
        assert en_correct('quintessential')[0] == 'quintessential'  # unknow

        return 'unit_test pass'

    def test_tokenizer(self):
        """测试英文切词"""
        sent = "test is it."
        white_split = whitespace_tokenize(sent)
        print(white_split)
        assert white_split == ['test', 'is', 'it', '.']  # segment
        res = ['This', 'is', 'a', 'test', 'of', 'the', 'word', 'parser', '.', 'It', 'should', 'work', 'correctly',
               '!!!']
        self.assertEqual(whitespace_tokenize('This is a test of the word parser. It should work correctly!!!'), res)

    @staticmethod
    def spell_t(tests):
        """
        run en_correct(wrong) on all (right,wrong) pairs, and report result
        :param tests:
        :return:
        """
        import time
        start = time.time()
        good, unknown = 0, 0
        n = len(tests)
        en_correct = EnSpell().correct
        for right, wrong in tests:
            w = en_correct(wrong)[0]
            good += (w == right)
        all_time = time.time() - start
        print('acc: {:.0%}, total num: {}, ({:.0%} unknown), speed: {:.0f} '
              'words per second'.format(good / n, n, unknown / n, n / all_time))

    @staticmethod
    def get_set(lines):
        """
        parse 'right, wrong1, wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs
        :param lines:
        :return:
        """
        return [(right, wrong) for (right, wrongs) in (line.split(':') for line in lines) for wrong in wrongs.split()]

    def test_spell1(self):
        """测试英文文本纠错-dev"""
        self.spell_t(self.get_set(open('./spell-testset1.txt')))  # Dev set

    def test_spell2(self):
        """测试英文文本纠错-test"""
        self.spell_t(self.get_set(open('./spell-testset2.txt')))  # final test set



if __name__ == '__main__':
    unittest.main()

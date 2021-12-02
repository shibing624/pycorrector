# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
import unittest

sys.path.append('..')
from pycorrector.en_spell import en_correct,EnSpell


class EnBugTestCase(unittest.TestCase):
    def test_en_bug_correct1(self):
        """测试英文纠错bug"""
        r = en_correct('folder payroll connectivity website')
        print(r)
        assert en_correct('spelling')[0] == 'spelling'  # no error

    def test_en_bug_correct2(self):
        """测试英文纠错bug"""
        spell = EnSpell()
        spell.check_init()
        print(spell.word_freq_dict.get('whould'))
        print(spell.candidates('whould'))

        a = spell.correct_word('whould')
        print(a)
        r = en_correct('contend proble poety adress whould niether  quaties')
        print(r)
        assert en_correct('whould')[0] == 'would'  # no error


if __name__ == '__main__':
    unittest.main()

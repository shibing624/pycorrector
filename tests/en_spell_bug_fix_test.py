# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys
import unittest

sys.path.append('..')
from pycorrector.en_spell import EnSpell

spell = EnSpell()
spell.check_init()


class EnBugTestCase(unittest.TestCase):
    def test_en_bug_correct1(self):
        """测试英文纠错bug"""
        r = spell.correct('folder payroll connectivity website')
        print(r)
        assert spell.correct('spelling')[0] == 'spelling'  # no error

    def test_en_bug_correct2(self):
        """测试英文纠错bug"""

        print(spell.word_freq_dict.get('whould'))
        print(spell.candidates('whould'))

        a = spell.correct_word('whould')
        print(a)
        r = spell.correct('contend proble poety adress whould niether  quaties')
        print(r)
        assert spell.correct('whould')[0] == 'would'  # no error


if __name__ == '__main__':
    unittest.main()

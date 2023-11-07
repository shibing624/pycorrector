# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest

from pycorrector import EnSpellCorrector

spell = EnSpellCorrector()


class TestEnSpell(unittest.TestCase):
    """ test the spell checker class """

    def test_correct(self):
        """ test spell checker corrects """
        self.assertEqual(spell.correct('ths')['target'], 'the')
        self.assertEqual(spell.correct('ergo')['target'], 'ergo')
        self.assertEqual(spell.correct('this')['target'], 'this')
        self.assertEqual(spell.correct('-')['target'], '-')
        self.assertEqual(spell.correct('1213')['target'], '1213')
        self.assertEqual(''.join(spell.correct('1213.9')['target']), '1213.9')

    def test_candidates(self):
        """ test spell checker candidates """
        print(spell.word_freq_dict.get('ths'), spell.candidates('ths'))
        self.assertEqual(len(spell.candidates('ths')) > 0, True)
        self.assertEqual(spell.candidates('the'), {'the'})
        self.assertEqual(spell.candidates('hi'), {'hi'})
        # something that cannot exist... should return just the same element...
        self.assertEqual(''.join(spell.candidates('manasaeds')), 'manasaeds')

    def test_word_frequency(self):
        """ test word frequency """
        # if the default load changes so will this...
        self.assertEqual(spell.word_freq_dict.get('he'), 12846723)

    def test_word_known(self):
        """ test if the word is a `known` word or not """
        self.assertEqual(spell.known(['this']), {'this'})
        self.assertEqual(spell.known(['hi']), {'hi'})
        self.assertEqual(spell.known(['holmes']), {'holmes'})
        self.assertEqual(spell.known(['known']), {'known'})

        self.assertEqual(spell.known(['-']), set())
        self.assertEqual(spell.known(['foobar']), set())
        self.assertEqual(spell.known(['ths']), set())
        self.assertEqual(spell.known(['ergos']), set())

    def test_word_in(self):
        """ test the use of the `in` operator """
        self.assertTrue('key' in spell.word_freq_dict)
        self.assertFalse('wantthis' in spell.word_freq_dict)  # a known excluded word
        self.assertEqual(spell.word_freq_dict.get('a', 0), 48779620)

    def test_case_insensitive_parse_words(self):
        """ Test using the parse words to generate dict """
        print(spell.word_freq_dict.get('thisss', 0))
        assert spell.word_freq_dict.get('thisss', 0) == 0

        dic = {'thisss': 1, 'is': 1, 'a': 1, 'test': 2, 'of': 1, 'the': 1}
        spell_new = EnSpellCorrector(word_freq_dict=dic)
        print(spell_new.word_freq_dict.get('thisss', 0))
        # in makes sure it is lower case in this instance
        self.assertTrue(spell_new.word_freq_dict.get('thisss', 0) == 1)


if __name__ == '__main__':
    unittest.main()

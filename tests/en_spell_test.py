# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
import sys

sys.path.append("../")
from pycorrector.en_spell import en_correct, words, spell, en_probability


def en_correct_t():
    assert en_correct('spelling') == 'spelling'  # no error
    assert en_correct('speling') == 'spelling'  # insert
    assert en_correct('correctud') == 'corrected'  # replace 1
    assert en_correct('gorrectud') == 'corrected'  # replace 2
    assert en_correct('bycycle') == 'bicycle'  # replace
    assert en_correct('inconvient') == 'inconvenient'  # insert 2
    assert en_correct('arrainged') == 'arranged'  # delete
    assert en_correct('peotrry') == 'poetry'  # transpose + delete
    assert en_correct('word') == 'word'  # know
    assert en_correct('quintessential') == 'quintessential'  # unknow
    assert words('the test is it.') == ['the', 'test', 'is', 'it']  # segment
    assert len(spell.WORDS) > 100
    assert spell.WORDS['the'] > 100
    assert en_probability('word') > 0
    assert en_probability('quintessential') == 0
    assert 0.07 < en_probability('the') < 0.08
    return 'unit_test pass'


def spell_t(tests, verbose=False):
    """
    run en_correct(wrong) on all (right,wrong) pairs, and report result
    :param tests:
    :param verbose:
    :return:
    """
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = en_correct(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in spell.WORDS)
            if verbose:
                print('en_correct({}) => {} ({}); expected {} ({})'.format(wrong, w, spell.WORDS[w], right,
                                                                           spell.WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second'.format(good / n, n, unknown / n, n / dt))


def get_set(lines):
    """
    parse 'right, wrong1, wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs
    :param lines:
    :return:
    """
    return [(right, wrong) for (right, wrongs) in (line.split(':') for line in lines) for wrong in wrongs.split()]


if __name__ == '__main__':
    print(en_correct_t())
    spell_t(get_set(open('../pycorrector/data/en/spell-testset1.txt')), verbose=True)  # Dev set
    spell_t(get_set(open('../pycorrector/data/en/spell-testset2.txt')), verbose=True)  # final test set

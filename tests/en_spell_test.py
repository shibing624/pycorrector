# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys
sys.path.append("../")
from pycorrector.en_spell import *

def correction_t():
    assert correction('spelling') == 'spelling'  # no error
    assert correction('speling') == 'spelling'  # insert
    assert correction('correctud') == 'corrected'  # replace 1
    assert correction('gorrectud') == 'corrected'  # replace 2
    assert correction('bycycle') == 'bicycle'  # replace
    assert correction('inconvient') == 'inconvenient'  # insert 2
    assert correction('arrainged') == 'arranged'  # delete
    assert correction('peotrry') == 'poetry'  # transpose + delete
    assert correction('word') == 'word'  # know
    assert correction('quintessential') == 'quintessential'  # unknow
    assert words('the test is it.') == ['the', 'test', 'is', 'it']  # segment
    assert len(WORDS) > 100
    assert WORDS['the'] > 100
    assert P('word') > 0
    assert P('quintessential') == 0
    assert 0.07 < P('the') < 0.08
    return 'unit_test pass'


def spell_t(tests, verbose=False):
    """
    run correction(wrong) on all (right,wrong) pairs, and report result
    :param tests:
    :param verbose:
    :return:
    """
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'.format(wrong, w, WORDS[w], right, WORDS[right]))
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
    print(correction_t())
    spell_t(get_set(open('../pycorrector/data/en/spell-testset1.txt')),verbose=True)  # Dev set
    spell_t(get_set(open('../pycorrector/data/en/spell-testset2.txt')),verbose=True)  # final test set

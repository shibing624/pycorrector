# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: english correction
# refer to http://norvig.com/spell-correct.html
import re
import os
from collections import Counter


def words(text):
    return re.findall(r'\w+', text.lower())


pwd_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(pwd_path, '../data/en/big.txt')
WORDS = Counter(words(open(path).read()))


def P(word, N=sum(WORDS.values())):
    """
    probability of word
    :param word:
    :param N:
    :return:
    """
    return WORDS[word] / N


def correction(word):
    """
    most probable spelling correction for word
    :param word:
    :return:
    """
    return max(candidates(word), key=P)


def candidates(word):
    """
    generate possible spelling corrections for word.
    :param word:
    :return:
    """
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words):
    """
    the subset of 'words' that appear in the dictionary of WORDS
    :param words:
    :return:
    """
    return set(w for w in words if w in WORDS)


def edits1(word):
    """
    all edits that are one edit away from 'word'
    :param word:
    :return:
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    all edit that are two edits away from 'word'
    :param word:
    :return:
    """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


if __name__ == '__main__':
    c1 = correction('speling')
    print(c1)
    c2 = correction('gorrect')
    print(c2)
    comm = WORDS.most_common(10)
    print(comm)
    max_word = max(WORDS, key=P)
    print(max_word)
    print(P('speling'))

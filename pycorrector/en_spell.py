# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: english correction
# refer to http://norvig.com/spell-correct.html
import os
import re
from collections import Counter

from pycorrector import config
from pycorrector.utils.logger import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))


def words(text):
    return re.findall(r'\w+', text.lower())


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


class EnSpell(object):
    def __init__(self, path=config.en_text_path):
        self.path = path
        self.WORDS = None

    def _init(self):
        self.WORDS = Counter(words(open(self.path).read()))
        logger.debug("load en spell data: %s, size: %d" % (self.path, len(self.WORDS)))

    def check_init(self):
        if not self.WORDS:
            self._init()

    def probability(self, word):
        """
        probability of word
        :param word:
        :return:float
        """
        self.check_init()
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correct(self, word):
        """
        most probable spelling correction for word
        :param word:
        :return:
        """
        self.check_init()
        return max(self.candidates(word), key=self.probability)

    def candidates(self, word):
        """
        generate possible spelling corrections for word.
        :param word:
        :return:
        """
        return self.known([word]) or self.known(edits1(word)) or self.known(edits2(word)) or [word]

    def known(self, words):
        """
        the subset of 'words' that appear in the dictionary of WORDS
        :param words:
        :return:
        """
        self.check_init()
        return set(w for w in words if w in self.WORDS)


spell = EnSpell()
en_correct = spell.correct
en_probability = spell.probability

if __name__ == '__main__':
    c1 = en_correct('speling')
    print(c1)
    c2 = en_correct('gorrect')
    print(c2)
    print(en_probability('speling'))
    erros = ['something', 'is', 'hapenning', 'here']
    for i in erros:
        print(en_correct(i))

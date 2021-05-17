# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: english correction
# refer to http://norvig.com/spell-correct.html

import gzip
import json
import operator
from collections import Counter

from pycorrector import config
from pycorrector.utils.logger import logger
from pycorrector.utils.tokenizer import whitespace_tokenize


def get_word_freq_dict_from_text(text):
    return Counter(whitespace_tokenize(text))


class EnSpell(object):
    def __init__(self, word_freq_dict={}):
        # Word freq dict, k=word, v=int(freq)
        self.word_freq_dict = word_freq_dict

    def _init(self):
        with gzip.open(config.en_dict_path, "rb") as f:
            all_word_freq_dict = json.loads(f.read(), encoding="utf-8")
            word_freq = {}
            for k, v in all_word_freq_dict.items():
                # 英语常用单词3万个，取词频高于400
                if v > 400:
                    word_freq[k] = v
            self.word_freq_dict = word_freq
            logger.debug("load en spell data: %s, size: %d" % (config.en_dict_path,
                                                               len(self.word_freq_dict)))

    def check_init(self):
        if not self.word_freq_dict:
            self._init()

    @staticmethod
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

    def edits2(self, word):
        """
        all edit that are two edits away from 'word'
        :param word:
        :return:
        """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, word_freq_dict, limit_count=500):
        """
        the subset of 'word_freq_dict' that appear in the dictionary of word_freq_dict
        :param word_freq_dict:
        :param limit_count:
        :return:
        """
        self.check_init()
        return set(w for w in word_freq_dict if w in self.word_freq_dict)

    def probability(self, word):
        """
        probability of word
        :param word:
        :return:float
        """
        self.check_init()
        N = sum(self.word_freq_dict.values())
        return self.word_freq_dict.get(word, 0) / N

    def candidates(self, word):
        """
        generate possible spelling corrections for word.
        :param word:
        :return:
        """
        self.check_init()
        return self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or {word}

    def correct_word(self, word):
        """
        most probable spelling correction for word
        :param word:
        :param mini_prob:
        :return:
        """
        self.check_init()
        candi_prob = {i: self.probability(i) for i in self.candidates(word)}
        sort_candi_prob = sorted(candi_prob.items(), key=operator.itemgetter(1))
        return sort_candi_prob[-1][0]

    def correct(self, text):
        """
        most probable spelling correction for text
        :param text:
        :return:
        """
        self.check_init()
        tokens = whitespace_tokenize(text)
        res = [self.correct_word(w) if len(w) > 1 else w for w in tokens]
        return res


spell = EnSpell()
en_correct = spell.correct
en_probability = spell.probability

if __name__ == '__main__':
    c1 = en_correct('speling is herr. do you know! !')
    print(c1)
    c2 = en_correct('gorrect')
    print(c2)
    print(en_probability('speling'))
    errors = ['something', 'is', 'hapenning', 'here']
    for i in errors:
        print(en_correct(i))
    sent = 'something is happending here, i konw!'
    print(sent, en_correct(sent))

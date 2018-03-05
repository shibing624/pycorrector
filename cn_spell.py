# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
# correct error based on word, refer to en_spell
import codecs
import os

from pypinyin import lazy_pinyin

from util import segment

pwd_path = os.path.abspath(os.path.dirname(__file__))
word_file_path = os.path.join(pwd_path, 'data/cn/word_dict.txt')
char_file_path = os.path.join(pwd_path, 'data/cn/char_set.txt')
PUNCTUATION_LIST = "。，,、？：；{}【】“‘’”《》/！%……（）"


def construct_dict(path):
    word_freq = {}
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            info = line.split()
            word = info[0]
            freq = info[1]
            word_freq[word] = freq
    return word_freq


phrase_freq = construct_dict(word_file_path)


def load_word_dict(path):
    word_dict = ''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            word_dict += w.strip()
    return word_dict


def edit1(phrase, word_dict):
    """
    all edits that are one edit away from 'phrase'
    :param phrase:
    :param word_dict:
    :return:
    """
    splits = [(phrase[:i], phrase[i:]) for i in range(len(phrase) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in word_dict]
    inserts = [L + c + R for L, R in splits for c in word_dict]
    return set(deletes + transposes + replaces + inserts)


def known(phrases):
    return set(phrase for phrase in phrases if phrase in phrase_freq)


def candidates(phrase):
    candidates_1_order = []
    candidates_2_order = []
    candidates_3_order = []
    error_pinyin = lazy_pinyin(phrase)
    cn_char_set = load_word_dict(char_file_path)
    candidate_phrases = list(known(edit1(phrase, cn_char_set)))
    for candidate_phrase in candidate_phrases:
        candidata_pinyin = lazy_pinyin(candidate_phrase)
        if candidata_pinyin == error_pinyin:
            candidates_1_order.append(candidate_phrase)
        elif candidata_pinyin[0] == error_pinyin[0]:
            candidates_2_order.append(candidate_phrase)
        else:
            candidates_3_order.append(candidate_phrase)
    return candidates_1_order, candidates_2_order, candidates_3_order


def correct_phrase(phrase):
    c1_order, c2_order, c3_order = candidates(phrase)
    if c1_order:
        return max(c1_order, key=phrase_freq.get)
    elif c2_order:
        return max(c2_order, key=phrase_freq.get)
    else:
        return max(c3_order, key=phrase_freq.get)


def correct(sentence, verbose=True):
    seg_list = segment(sentence)
    correct_sentence = ''
    for phrase in seg_list:
        corrected_phrase = phrase
        if phrase not in PUNCTUATION_LIST:
            if phrase not in phrase_freq.keys():
                corrected_phrase = correct_phrase(phrase)
                if verbose:
                    print(phrase, '=>', corrected_phrase)
        correct_sentence += corrected_phrase
    return correct_sentence

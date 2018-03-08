# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
# correct error based on word, refer to en_spell
import codecs
import os

from pypinyin import lazy_pinyin

from pycorrector.util import segment

pwd_path = os.path.abspath(os.path.dirname(__file__))
word_file_path = os.path.join(pwd_path, 'data/cn/word_dict.txt')
char_file_path = os.path.join(pwd_path, 'data/cn/char_set.txt')
PUNCTUATION_LIST = "。，,、？：；{}[]【】“‘’”《》/！%……（）<>@#$~^￥%&*\"\'=+-"


def construct_dict(path):
    word_freq = {}
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            info = line.split()
            word = info[0]
            freq = int(info[1])
            word_freq[word] = freq
    return word_freq


word_freq = construct_dict(word_file_path)


def load_word_dict(path):
    word_dict = ''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            word_dict += w.strip()
    return word_dict


def edit1(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    inserts = [L + c + R for L, R in splits for c in char_set]
    return set(deletes + transposes + replaces + inserts)


def edit_distance_word(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    return set(transposes + replaces)


def known(words):
    return set(word for word in words if word in word_freq)


def candidates(word):
    candidates_1_order = []
    candidates_2_order = []
    candidates_3_order = []
    error_pinyin = lazy_pinyin(word)
    cn_char_set = load_word_dict(char_file_path)
    candidate_words = list(known(edit_distance_word(word, cn_char_set)))
    for candidate_word in candidate_words:
        candidata_pinyin = lazy_pinyin(candidate_word)
        if candidata_pinyin == error_pinyin:
            candidates_1_order.append(candidate_word)
        elif candidata_pinyin[0] == error_pinyin[0]:
            candidates_2_order.append(candidate_word)
        else:
            candidates_3_order.append(candidate_word)
    return candidates_1_order, candidates_2_order, candidates_3_order


def correct_word(word):
    c1_order, c2_order, c3_order = candidates(word)
    if c1_order:
        return max(c1_order, key=word_freq.get)
    elif c2_order:
        return max(c2_order, key=word_freq.get)
    elif c3_order:
        return max(c3_order, key=word_freq.get)
    else:
        return word


def correct(sentence, verbose=False):
    """
    correct the error sentence to correct sentence
    :param sentence: str, input sentence with error words
    :param verbose: bool,
    :return: correct_sentence, correct_detail
    """
    correct_sentence = ''
    wrong_words, right_words, begin_idx, end_idx = [], [], [], []
    seg_words = segment(sentence)
    for word in seg_words:
        corrected_word = word
        if word not in PUNCTUATION_LIST:
            if word not in word_freq.keys():
                corrected_word = correct_word(word)
                begin_index = sentence.find(word)
                begin_idx.append(begin_index)
                end_idx.append(begin_index + len(word))
                wrong_words.append(word)
                right_words.append(corrected_word)
                if verbose:
                    print('pred:', word, '=>', corrected_word)
        correct_sentence += corrected_word
    correct_detail = list(zip(wrong_words, right_words, begin_idx, end_idx))
    return correct_sentence, correct_detail

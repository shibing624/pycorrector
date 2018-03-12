# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: corrector with spell and stroke

import codecs
import os

import pypinyin
from pypinyin import pinyin, lazy_pinyin

from pycorrector.detector import detect, trigram_char
from pycorrector.detector import get_frequency, word_freq, get_ppl_score
from pycorrector.util import dump_pkl
from pycorrector.util import load_pkl

pwd_path = os.path.abspath(os.path.dirname(__file__))
char_file_path = os.path.join(pwd_path, 'data/cn/char_set.txt')


def load_word_dict(path):
    word_dict = ''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            word_dict += w.strip()
    return word_dict


def load_same_pinyin(path, sep='\t'):
    """
    加载同音字
    :param path:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        print("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                same_pron_same_tone = set(list(parts[1]))
                same_pron_diff_tone = set(list(parts[2]))
                value = same_pron_same_tone.union(same_pron_diff_tone)
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value
    return result


def load_same_stroke(path, sep='\t'):
    """
    加载形似字
    :param path:
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        print("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 1:
                key_char = parts[0]
                result[key_char] = set(list(parts[1]))
    return result


same_pinyin_text_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
same_pinyin_model_path = os.path.join(pwd_path, 'data/same_pinyin.pkl')
# 同音字
if os.path.exists(same_pinyin_model_path):
    same_pinyin = load_pkl(same_pinyin_model_path)
else:
    print('load same pinyin from text file:', same_pinyin_text_path)
    same_pinyin = load_same_pinyin(same_pinyin_text_path)
    dump_pkl(same_pinyin, same_pinyin_model_path)

# 形似字
same_stroke_text_path = os.path.join(pwd_path, 'data/same_stroke.txt')
same_stroke_model_path = os.path.join(pwd_path, 'data/same_stroke.pkl')
if os.path.exists(same_stroke_model_path):
    same_stroke = load_pkl(same_stroke_model_path)
else:
    print('load same stroke from text file:', same_stroke_text_path)
    same_stroke = load_same_stroke(same_stroke_text_path)
    dump_pkl(same_stroke, same_stroke_model_path)


def get_same_pinyin(char):
    """
    取同音字
    :param char:
    :return:
    """
    return same_pinyin.get(char, set())


def get_same_stroke(char):
    """
    取形似字
    :param char:
    :return:
    """
    return same_stroke.get(char, set())


def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result


def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result


def generate_chars(c, fraction=2):
    """
    取音似、形似字
    :param c:
    :param fraction:
    :return:
    """
    confusion_char_set = get_same_pinyin(c)
    # confusion_char_set = get_same_pinyin(c).union(get_same_stroke(c))
    if not confusion_char_set:
        confusion_char_set = {c}
    confusion_char_set.add(c)
    confusion_char_list = list(confusion_char_set)
    all_confusion_char = sorted(confusion_char_list, key=lambda k: \
        get_frequency(k), reverse=True)
    return all_confusion_char[:len(confusion_char_list) // fraction + 1]


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


def _known(words):
    return set(word for word in words if word in word_freq)


def _candidates(word):
    candidates_1_order = []
    candidates_2_order = []
    candidates_3_order = []
    error_pinyin = lazy_pinyin(word)
    cn_char_set = load_word_dict(char_file_path)
    candidate_words = list(_known(edit_distance_word(word, cn_char_set)))
    for candidate_word in candidate_words:
        candidata_pinyin = lazy_pinyin(candidate_word)
        if candidata_pinyin == error_pinyin:
            candidates_1_order.append(candidate_word)
        elif candidata_pinyin[0] == error_pinyin[0]:
            candidates_2_order.append(candidate_word)
        else:
            candidates_3_order.append(candidate_word)
    return candidates_1_order, candidates_2_order, candidates_3_order


def _correct_word(word):
    c1_order, c2_order, c3_order = _candidates(word)
    if c1_order:
        return max(c1_order, key=word_freq.get)
    elif c2_order:
        return max(c2_order, key=word_freq.get)
    elif c3_order:
        return max(c3_order, key=word_freq.get)
    else:
        return word


def _correct_chars(sentence, ids):
    """
    纠正错字，逐字处理
    :param sentence:
    :param ids:
    :return: corrected characters 修正的汉字
    """
    wrongs, rights, error_ids = [], [], []
    maybe_error_chars = {i: sentence[i] for i in ids}
    corrected_sent = sentence
    for i, c in maybe_error_chars.items():
        # 取得所有可能正确的汉字
        maybe_chars = generate_chars(c)
        # print('num of possible replacements for {} is {}'.format(c, len(maybe_chars)))
        before = corrected_sent[:i]
        after = corrected_sent[i + 1:]
        corrected_c = min(maybe_chars,
                          key=lambda k: get_ppl_score(list(before + k + after),
                                                      mode=trigram_char))
        if corrected_c != c:
            error_ids.append(i)
            corrected_sent = before + corrected_c + after
            print('pred:', c, '=>', corrected_c)
            wrongs.append(c)
            rights.append(corrected_c)
    detail = list(zip(wrongs, rights, error_ids))
    return corrected_sent, detail


def correct(sentence):
    """
    句子改错
    :param sentence: 句子文本
    :return: 改正后的句子, list(wrongs, rights, error_ids)
    """
    maybe_error_ids = detect(sentence)
    return _correct_chars(sentence, maybe_error_ids)


if __name__ == '__main__':
    line = '少先队员因该为老人让坐'
    print('input sentence is:', line)
    corrected_sent, detail = correct(line)
    print('corrected_sent:', corrected_sent)
    print('detail:', detail)

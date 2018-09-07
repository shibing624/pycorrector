# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: corrector with spell and stroke
import codecs
import os
import time

from pypinyin import lazy_pinyin

from pycorrector.detector import Detector
from pycorrector.utils.io_utils import get_logger
from pycorrector.utils.math_utils import edit_distance_word, get_sub_array
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.text_utils import traditional2simplified

default_logger = get_logger(__file__)
pwd_path = os.path.abspath(os.path.dirname(__file__))


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
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        default_logger.warn("file not exists:" + path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = traditional2simplified(line.strip())
            parts = line.split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                same_pron_same_tone = set(list(parts[1]))
                same_pron_diff_tone = set(list(parts[2]))
                value = same_pron_same_tone.union(same_pron_diff_tone)
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value
    return result


def load_same_stroke(path, sep=','):
    """
    加载形似字
    :param path:
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        default_logger.warn("file not exists:" + path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = traditional2simplified(line.strip())
            parts = line.strip().split(sep)
            if parts and len(parts) > 1:
                for i, c in enumerate(parts):
                    result[c] = set(list(parts[:i] + parts[i + 1:]))
    return result


class Corrector(object):
    def __init__(self, char_file_path='', same_pinyin_text_path='',
                 same_stroke_text_path='', language_model_path='', word_freq_path=''):
        self.char_file_path = os.path.join(pwd_path, char_file_path)
        self.same_pinyin_text_path = os.path.join(pwd_path, same_pinyin_text_path)
        self.same_stroke_text_path = os.path.join(pwd_path, same_stroke_text_path)
        self.detector = Detector(language_model_path=language_model_path, word_freq_path=word_freq_path)
        self.initialized = False

    def initialize(self):
        t1 = time.time()
        # chinese common char dict
        self.cn_char_set = load_word_dict(self.char_file_path)
        # same pinyin
        self.same_pinyin = load_same_pinyin(self.same_pinyin_text_path)
        # same stroke
        self.same_stroke = load_same_stroke(self.same_stroke_text_path)
        default_logger.debug("Loaded same pinyin file: %s, same stroke file: %s, spend: %.3f s." % (
            self.same_pinyin_text_path, self.same_stroke_text_path, time.time() - t1))
        self.initialized = True

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def get_same_pinyin(self, char):
        """
        取同音字
        :param char:
        :return:
        """
        self.check_initialized()
        return self.same_pinyin.get(char, set())

    def get_same_stroke(self, char):
        """
        取形似字
        :param char:
        :return:
        """
        self.check_initialized()
        return self.same_stroke.get(char, set())

    def known(self, words):
        return set(word for word in words if word in self.detector.word_freq)

    def _confusion_char_set(self, c):
        confusion_char_set = self.get_same_pinyin(c).union(self.get_same_stroke(c))
        if not confusion_char_set:
            confusion_char_set = set()
        return confusion_char_set

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        candidate_words = list(self.known(edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            if lazy_pinyin(candidate_word) == lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _generate_items(self, word, fraction=1):
        candidates_1_order = []
        candidates_2_order = []
        candidates_3_order = []
        # same pinyin word
        candidates_1_order.extend(self._confusion_word_set(word))
        # same pinyin char
        if len(word) == 1:
            # same pinyin
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_2_order.extend(confusion)
        if len(word) > 1:
            # same first pinyin
            confusion = [i + word[1:] for i in self._confusion_char_set(word[0]) if i]
            candidates_2_order.extend(confusion)
            # same last pinyin
            confusion = [word[:-1] + i for i in self._confusion_char_set(word[-1]) if i]
            candidates_2_order.extend(confusion)
            if len(word) > 2:
                # same mid char pinyin
                confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
                candidates_3_order.extend(confusion)

                # same first word pinyin
                confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
                candidates_1_order.extend(confusion_word)

                # same last word pinyin
                confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
                candidates_1_order.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1_order + candidates_2_order + candidates_3_order)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k:
        self.detector.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fraction + 1]

    def _correct_item(self, sentence, idx, item):
        """
        纠正错误，逐词处理
        :param sentence:
        :param idx:
        :param item:
        :return: corrected word 修正的词语
        """
        corrected_sent = sentence
        if not is_chinese_string(item):
            return corrected_sent, []
        # 取得所有可能正确的词
        maybe_error_items = self._generate_items(item)
        if not maybe_error_items:
            return corrected_sent, []
        ids = idx.split(',')
        begin_id = int(ids[0])
        end_id = int(ids[-1]) if len(ids) > 1 else int(ids[0]) + 1
        before = sentence[:begin_id]
        after = sentence[end_id:]
        corrected_item = min(maybe_error_items,
                             key=lambda k: self.detector.ppl_score(list(before + k + after)))
        wrongs, rights, begin_idx, end_idx = [], [], [], []
        if corrected_item != item:
            corrected_sent = before + corrected_item + after
            # default_logger.debug('pred:', item, '=>', corrected_item)
            wrongs.append(item)
            rights.append(corrected_item)
            begin_idx.append(begin_id)
            end_idx.append(end_id)
        detail = list(zip(wrongs, rights, begin_idx, end_idx))
        return corrected_sent, detail

    def correct(self, sentence):
        """
        句子改错
        :param sentence: 句子文本
        :return: 改正后的句子, list(wrongs, rights, begin_idx, end_idx)
        """
        self.check_initialized()
        detail = []
        maybe_error_ids = get_sub_array(self.detector.detect(sentence))
        # print('maybe_error_ids:', maybe_error_ids)
        # 取得字词对应表
        index_char_dict = dict()
        for index in maybe_error_ids:
            if len(index) == 1:
                # 字
                index_char_dict[','.join(map(str, index))] = sentence[index[0]]
            else:
                # 词
                index_char_dict[','.join(map(str, index))] = sentence[index[0]:index[-1]]
        for index, item in index_char_dict.items():
            # 字词纠错
            sentence, detail_word = self._correct_item(sentence, index, item)
            if detail_word:
                detail.append(detail_word)
        return sentence, detail

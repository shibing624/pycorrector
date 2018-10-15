# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: corrector with spell and stroke
import codecs
import os
import time

from pypinyin import lazy_pinyin

from pycorrector.detector import Detector, error_type
from pycorrector.utils.io_utils import get_logger
from pycorrector.utils.math_utils import edit_distance_word
from pycorrector.utils.text_utils import is_chinese_string

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
            line = line.strip()
            if line.startswith('#'):
                continue
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


def load_same_stroke(path, sep='\t'):
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
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split(sep)
            if parts and len(parts) > 1:
                for i, c in enumerate(parts):
                    result[c] = set(list(parts[:i] + parts[i + 1:]))
    return result


class Corrector(Detector):
    def __init__(self, common_char_path='', same_pinyin_path='',
                 same_stroke_path='', language_model_path='',
                 word_freq_path='', custom_confusion_path=''):
        super(Corrector, self).__init__(language_model_path=language_model_path,
                                        word_freq_path=word_freq_path,
                                        custom_confusion_path=custom_confusion_path)
        self.name = 'corrector'
        self.common_char_path = os.path.join(pwd_path, common_char_path)
        self.same_pinyin_text_path = os.path.join(pwd_path, same_pinyin_path)
        self.same_stroke_text_path = os.path.join(pwd_path, same_stroke_path)
        self.initialized_corrector = False

    def initialize_corrector(self):
        t1 = time.time()
        # chinese common char dict
        self.cn_char_set = load_word_dict(self.common_char_path)
        # same pinyin
        self.same_pinyin = load_same_pinyin(self.same_pinyin_text_path)
        # same stroke
        self.same_stroke = load_same_stroke(self.same_stroke_text_path)
        default_logger.debug("Loaded same pinyin file: %s, same stroke file: %s, spend: %.3f s." % (
            self.same_pinyin_text_path, self.same_stroke_text_path, time.time() - t1))
        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self.initialize_corrector()

    def get_same_pinyin(self, char):
        """
        取同音字
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_pinyin.get(char, set())

    def get_same_stroke(self, char):
        """
        取形似字
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_stroke.get(char, set())

    def known(self, words):
        return set(word for word in words if word in self.word_freq)

    def _confusion_char_set(self, c):
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        candidate_words = list(self.known(edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            if lazy_pinyin(candidate_word) == lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _confusion_custom_set(self, word):
        confusion_word_set = set()
        if word in self.custom_confusion:
            confusion_word_set = {self.custom_confusion[word]}
        return confusion_word_set

    def _generate_items(self, word, fraction=1):
        candidates_1_order = []
        candidates_2_order = []
        candidates_3_order = []
        # same pinyin word
        candidates_1_order.extend(self._confusion_word_set(word))
        # custom confusion word
        candidates_1_order.extend(self._confusion_custom_set(word))
        # same pinyin char
        if len(word) == 1:
            # same one char pinyin
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_2_order.extend(confusion)
        if len(word) == 2:
            # same first char pinyin
            confusion = [i + word[1:] for i in self._confusion_char_set(word[0]) if i]
            candidates_2_order.extend(confusion)
            # same last char pinyin
            confusion = [word[:-1] + i for i in self._confusion_char_set(word[-1]) if i]
            candidates_2_order.extend(confusion)
        if len(word) > 2:
            # same mid char pinyin
            confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
            candidates_3_order.extend(confusion)

            # same first word pinyin
            confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
            candidates_3_order.extend(confusion_word)

            # same last word pinyin
            confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
            candidates_3_order.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1_order + candidates_2_order + candidates_3_order)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fraction + 1]

    def _correct_item(self, sentence, item, begin_idx, end_idx, err_type):
        """
        纠正字词错误
        :param sentence:
        :param item:
        :param begin_idx:
        :param end_idx:
        :param err_type: 错误类型
        :return: corrected word 修正的词语
        """
        corrected_sent = sentence
        detail = []
        if not is_chinese_string(item):
            return corrected_sent, detail
        # 取得所有可能正确的词
        maybe_right_items = self._generate_items(item)
        if not maybe_right_items:
            return corrected_sent, []
        before_sent = sentence[:begin_idx]
        after_sent = sentence[end_idx:]
        if err_type == error_type["confusion"]:
            corrected_item = self.custom_confusion[item]
        else:
            corrected_item = min(maybe_right_items, key=lambda k: self.ppl_score(list(before_sent + k + after_sent)))
        if corrected_item != item:
            corrected_sent = before_sent + corrected_item + after_sent
            # default_logger.debug('predict:' + item + '=>' + corrected_item)
            detail = [item, corrected_item, begin_idx, end_idx]
        return corrected_sent, detail

    def correct(self, sentence):
        """
        句子改错
        :param sentence: 句子文本
        :return: 改正后的句子, list(wrong, right, begin_idx, end_idx)
        """
        detail = []
        self.check_corrector_initialized()
        maybe_errors = self.detect(sentence)
        for item, begin_idx, end_idx, err_type in maybe_errors:
            # 纠错，逐个处理
            sentence, detail_word = self._correct_item(sentence, item, begin_idx, end_idx, err_type)
            if detail_word:
                detail.append(detail_word)
        return sentence, detail

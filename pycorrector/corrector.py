# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: corrector with pinyin and stroke
"""
import operator
import os
from codecs import open
from typing import List

import pypinyin
from loguru import logger

from pycorrector.detector import Detector, ErrorType
from pycorrector.utils.math_utils import edit_distance_word
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.tokenizer import segment, split_text_into_sentences_by_symbol

pwd_path = os.path.abspath(os.path.dirname(__file__))

# 中文常用字符集
common_char_path = os.path.join(pwd_path, 'data/common_char_set.txt')
# 同音字
same_pinyin_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
# 形似字
same_stroke_path = os.path.join(pwd_path, 'data/same_stroke.txt')


class Corrector(Detector):
    def __init__(
            self,
            common_char_path=common_char_path,
            same_pinyin_path=same_pinyin_path,
            same_stroke_path=same_stroke_path,
            **kwargs,
    ):
        super(Corrector, self).__init__(**kwargs)
        self.name = 'corrector'
        self.common_char_path = common_char_path
        self.same_pinyin_path = same_pinyin_path
        self.same_stroke_path = same_stroke_path
        self.initialized_corrector = False
        self.cn_char_set = None
        self.same_pinyin = None
        self.same_stroke = None

    @staticmethod
    def load_set_file(path):
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_same_pinyin(path, sep='\t'):
        """
        加载同音字
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warning(f"file not exists: {path}")
            return result
        with open(path, 'r', encoding='utf-8') as f:
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
                    if key_char and value:
                        result[key_char] = value
        return result

    @staticmethod
    def load_same_stroke(path, sep='\t'):
        """
        加载形似字
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warning(f"file not exists: {path}")
            return result
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        exist = result.get(c, set())
                        current = set(list(parts[:i] + parts[i + 1:]))
                        result[c] = exist.union(current)
        return result

    def _initialize_corrector(self):
        # chinese common char
        self.cn_char_set = self.load_set_file(self.common_char_path)
        # same pinyin
        self.same_pinyin = self.load_same_pinyin(self.same_pinyin_path)
        # same stroke
        self.same_stroke = self.load_same_stroke(self.same_stroke_path)
        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self._initialize_corrector()

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
        """
        取得词序列中属于常用词部分
        :param words:
        :return:
        """
        self.check_detector_initialized()
        return set(word for word in words if word in self.word_freq)

    def _confusion_char_set(self, c):
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        candidate_words = list(self.known(edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            if pypinyin.lazy_pinyin(candidate_word) == pypinyin.lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _confusion_custom_set(self, word):
        confusion_word_set = set()
        if word in self.custom_confusion:
            confusion_word_set = {self.custom_confusion[word]}
        return confusion_word_set

    def generate_items(self, word, fragment=1):
        """
        生成纠错候选集
        :param word:
        :param fragment: 分段
        :return:
        """
        self.check_corrector_initialized()
        # 1字
        candidates_1 = []
        # 2字
        candidates_2 = []
        # 多于2字
        candidates_3 = []

        # same pinyin word
        candidates_1.extend(self._confusion_word_set(word))
        # custom confusion word
        candidates_1.extend(self._confusion_custom_set(word))
        # get similarity char
        if len(word) == 1:
            # sim one char
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_1.extend(confusion)
        if len(word) == 2:
            # sim first char
            confusion_first = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_2.extend([i + word[1] for i in confusion_first])
            # sim last char
            confusion_last = [i for i in self._confusion_char_set(word[1]) if i]
            candidates_2.extend([word[0] + i for i in confusion_last])
            # both change, sim char
            candidates_2.extend([i + j for i in confusion_first for j in confusion_last if i + j])
            # sim word
            # candidates_2.extend([i for i in self._confusion_word_set(word) if i])
        if len(word) > 2:
            # sim mid char
            confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
            candidates_3.extend(confusion)

            # sim first word
            confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
            candidates_3.extend(confusion_word)

            # sim last word
            confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
            candidates_3.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1 + candidates_2 + candidates_3)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fragment + 1]

    def get_lm_correct_item(self, cur_item, candidates, before_sent, after_sent, threshold=57.0, cut_type='char'):
        """
        通过语言模型纠正字词错误
        :param cur_item: 当前词
        :param candidates: 候选词
        :param before_sent: 前半部分句子
        :param after_sent: 后半部分句子
        :param threshold: ppl阈值, 原始字词替换后大于该ppl值则认为是错误
        :param cut_type: 切词方式, 字粒度
        :return: str, correct item, 正确的字词
        """
        result = cur_item
        if cur_item not in candidates:
            candidates.append(cur_item)

        ppl_scores = {i: self.ppl_score(segment(before_sent + i + after_sent, cut_type=cut_type)) for i in candidates}
        sorted_ppl_scores = sorted(ppl_scores.items(), key=lambda d: d[1])

        # 增加正确字词的修正范围，减少误纠
        top_items = []
        top_score = 0.0
        for i, v in enumerate(sorted_ppl_scores):
            v_word = v[0]
            v_score = v[1]
            if i == 0:
                top_score = v_score
                top_items.append(v_word)
            # 通过阈值修正范围
            elif v_score < top_score + threshold:
                top_items.append(v_word)
            else:
                break
        if cur_item not in top_items:
            result = top_items[0]
        return result

    def correct(
            self,
            sentence: str,
            include_symbol: bool = True,
            num_fragment: int = 1,
            threshold: float = 57.0,
            **kwargs
    ):
        """
        单条文本纠错

        纠错逻辑：
        1. 自定义混淆集
        2. 专名错误
        3. 字词错误
        :param sentence: str, query 文本
        :param include_symbol: bool, 是否包含标点符号
        :param num_fragment: 纠错候选集分段数, 1 / (num_fragment + 1)
        :param threshold: 语言模型纠错ppl阈值
        :param kwargs: ...
        :return: text (str)改正后的句子, list(wrong, right, begin_idx, end_idx)
        """
        corrected_sentence = ''
        details = []
        self.check_corrector_initialized()
        # 按标点符号切分短句
        short_sents = split_text_into_sentences_by_symbol(sentence, include_symbol=include_symbol)
        for sent, idx in short_sents:
            # 检错
            maybe_errors, proper_details = self._detect(sent, idx, **kwargs)
            for cur_item, begin_idx, end_idx, err_type in maybe_errors:
                # 纠错，逐个处理
                before_sent = sent[:(begin_idx - idx)]
                after_sent = sent[(end_idx - idx):]

                # 困惑集中指定的词，直接取结果
                if err_type == ErrorType.confusion:
                    corrected_item = self.custom_confusion[cur_item]
                elif err_type == ErrorType.proper:
                    # 专名错误 proper_details format: (error_word, corrected_word, begin_idx, end_idx)
                    corrected_item = [i[1] for i in proper_details if cur_item == i[0] and begin_idx == i[2]][0]
                else:
                    # 字词错误，找所有可能正确的词
                    candidates = self.generate_items(cur_item, fragment=num_fragment)
                    if not candidates:
                        continue
                    corrected_item = self.get_lm_correct_item(
                        cur_item,
                        candidates,
                        before_sent,
                        after_sent,
                        threshold=threshold
                    )
                # output
                if corrected_item != cur_item:
                    sent = before_sent + corrected_item + after_sent
                    detail_word = (cur_item, corrected_item, begin_idx, end_idx)
                    details.append(detail_word)
            corrected_sentence += sent
        details = sorted(details, key=operator.itemgetter(2))
        return corrected_sentence, details

    def correct_batch(self, sentences: List[str], **kwargs):
        """Correct sentences with correct method."""
        corrected_sentences = []
        corrected_details = []
        for sentence in sentences:
            corrected_sent, details = self.correct(sentence, **kwargs)
            corrected_sentences.append(corrected_sent)
            corrected_details.append(details)
        return corrected_sentences, corrected_details

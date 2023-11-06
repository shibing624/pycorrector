# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 专名纠错，包括成语纠错、人名纠错、机构名纠错、领域词纠错等
"""
import os
from codecs import open

import pypinyin
from loguru import logger

from pycorrector.utils.math_utils import edit_distance
from pycorrector.utils.ngram_util import NgramUtil
from pycorrector.utils.text_utils import is_chinese_char
from pycorrector.utils.tokenizer import segment, split_text_into_sentences_by_symbol

pwd_path = os.path.abspath(os.path.dirname(__file__))

# 五笔笔画字典
stroke_path = os.path.join(pwd_path, 'data/stroke.txt')
# 专名词典，包括成语、俗语、专业领域词等 format: 词语
proper_name_path = os.path.join(pwd_path, 'data/proper_name.txt')


def load_set_file(path):
    words = set()
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
    return words


def load_dict_file(path):
    """
    加载词典
    :param path:
    :return:
    """
    result = {}
    if path:
        if not os.path.exists(path):
            logger.warning('file not found.%s' % path)
            return result
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    terms = line.split()
                    if len(terms) < 2:
                        continue
                    result[terms[0]] = terms[1]
    return result


class ProperCorrector:
    def __init__(
            self,
            proper_name_path=proper_name_path,
            stroke_path=stroke_path,
    ):
        self.name = 'ProperCorrector'
        # proper name, 专名词典，包括成语、俗语、专业领域词等 format: 词语
        self.proper_names = load_set_file(proper_name_path)
        # stroke, 笔划字典 format: 字:笔划，如：万，笔划是横(h),折(z),撇(p),组合起来是：hzp
        self.stroke_dict = load_dict_file(stroke_path)

    def get_stroke(self, char):
        """
        取笔画
        :param char:
        :return:
        """
        return self.stroke_dict.get(char, '')

    def get_pinyin(self, char):
        return pypinyin.lazy_pinyin(char)

    def is_near_stroke_char(self, char1, char2, stroke_threshold=0.8):
        """
        判断两个字是否形似
        :param char1:
        :param char2:
        :return: bool
        """
        return self.get_char_stroke_similarity_score(char1, char2) > stroke_threshold

    def get_char_stroke_similarity_score(self, char1, char2):
        """
        获取字符的字形相似度
        Args:
            char1:
            char2:

        Returns:
            float, 字符相似度值
        """
        score = 0.0
        if char1 == char2:
            score = 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese_char(char1) != is_chinese_char(char2):
            return score
        if not is_chinese_char(char1):
            return score
        char_stroke1 = self.get_stroke(char1)
        char_stroke2 = self.get_stroke(char2)
        # 相似度计算：1-编辑距离
        score = 1.0 - edit_distance(char_stroke1, char_stroke2)
        return score

    def get_word_stroke_similarity_score(self, word1, word2):
        """
        计算两个词的字形相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        if word1 == word2:
            return 1.0
        if len(word1) != len(word2):
            return 0.0
        total_score = 0.0
        for i in range(len(word1)):
            char1 = word1[i]
            char2 = word2[i]
            if not self.is_near_stroke_char(char1, char2):
                return 0.0
            char_sim_score = self.get_char_stroke_similarity_score(char1, char2)
            total_score += char_sim_score
        score = total_score / len(word1)
        return score

    def is_near_pinyin_char(self, char1, char2) -> bool:
        """
        判断两个单字的拼音是否是临近读音
        :param char1:
        :param char2:
        :return: bool
        """
        char_pinyin1 = self.get_pinyin(char1)[0]
        char_pinyin2 = self.get_pinyin(char2)[0]
        if not char_pinyin1 or not char_pinyin2:
            return False
        if len(char_pinyin1) == len(char_pinyin2):
            return True
        confuse_dict = {
            "l": "n",
            "zh": "z",
            "ch": "c",
            "sh": "s",
            "eng": "en",
            "ing": "in",
        }
        for k, v in confuse_dict.items():
            if char_pinyin1.replace(k, v) == char_pinyin2.replace(k, v):
                return True
        return False

    def get_char_pinyin_similarity_score(self, char1, char2):
        """
        获取字符的拼音相似度
        :param char1:
        :param char2:
        :return: float, 相似度
        """
        score = 0.0
        if char1 == char2:
            score = 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese_char(char1) != is_chinese_char(char2):
            return score
        if not is_chinese_char(char1):
            return score
        char_pinyin1 = self.get_pinyin(char1)[0]
        char_pinyin2 = self.get_pinyin(char2)[0]
        # 相似度计算：1-编辑距离
        score = 1.0 - edit_distance(char_pinyin1, char_pinyin2)
        return score

    def get_word_pinyin_similarity_score(self, word1, word2):
        """
        计算两个词的拼音相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        if word1 == word2:
            return 1.0
        if len(word1) != len(word2):
            return 0.0
        total_score = 0.0
        for i in range(len(word1)):
            char1 = word1[i]
            char2 = word2[i]
            if not self.is_near_pinyin_char(char1, char2):
                return 0.0
            char_sim_score = self.get_char_pinyin_similarity_score(char1, char2)
            total_score += char_sim_score
        score = total_score / len(word1)
        return score

    def get_word_similarity_score(self, word1, word2):
        """
        计算两个词的相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        return max(
            self.get_word_stroke_similarity_score(word1, word2),
            self.get_word_pinyin_similarity_score(word1, word2)
        )

    def correct(
            self,
            sentence,
            start_idx=0,
            cut_type='char',
            ngram=1234,
            sim_threshold=0.85,
            max_word_length=4,
            min_word_length=2
    ):
        """
        专名纠错
        :param sentence: str, 待纠错的文本
        :param start_idx: int, 文本开始的索引，兼容correct方法
        :param cut_type: str, 分词类型，'char' or 'word'
        :param ngram: 遍历句子的ngram
        :param sim_threshold: 相似度得分阈值，超过该阈值才进行纠错
        :param max_word_length: int, 专名词的最大长度为4
        :param min_word_length: int, 专名词的最小长度为2
        :return: dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        text_new = ''
        details = []
        # 切分为短句
        sentences = split_text_into_sentences_by_symbol(sentence, include_symbol=True)
        for sentence, idx in sentences:
            # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
            sentence_words = segment(sentence, cut_type=cut_type)
            ngrams = NgramUtil.ngrams(sentence_words, ngram, join_string="_")
            # 去重
            ngrams = list(set([i.replace("_", "") for i in ngrams if i]))
            # 词长度过滤
            ngrams = [i for i in ngrams if min_word_length <= len(i) <= max_word_length]
            for cur_item in ngrams:
                for name in self.proper_names:
                    if self.get_word_similarity_score(cur_item, name) > sim_threshold:
                        if cur_item != name:
                            cur_idx = sentence.find(cur_item)
                            sentence = sentence[:cur_idx] + name + sentence[(cur_idx + len(cur_item)):]
                            details.append((cur_item, name, idx + cur_idx + start_idx))
            text_new += sentence
        return {'source': sentence, 'target': text_new, 'errors': details}

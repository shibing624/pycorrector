# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 专名纠错，包括成语纠错、人名纠错、机构名纠错、领域词纠错等
"""
import os
from codecs import open
from typing import Dict, List, Set
import pypinyin
from loguru import logger

from pycorrector.utils.math_utils import edit_distance
from pycorrector.utils.ngram_util import NgramUtil
from pycorrector.utils.text_utils import is_chinese_char
from pycorrector.utils.tokenizer import segment, split_text_into_sentences_by_symbol
from collections import defaultdict

pwd_path = os.path.abspath(os.path.dirname(__file__))
# 五笔笔画字典
stroke_path = os.path.join(pwd_path, 'data/stroke.txt')
# 专名词典，包括成语、俗语、专业领域词等 format: 词语, 可以自定义
default_proper_name_path = os.path.join(pwd_path, 'data/proper_name.txt')


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
            proper_name_path=default_proper_name_path,
            stroke_path=stroke_path,
    ):
        self.name = 'ProperCorrector'
        # proper name, 专名词典，包括成语、俗语、专业领域词等 format: 词语
        self.proper_names = load_set_file(proper_name_path)
        # stroke, 笔划字典 format: 字:笔划，如：万，笔划是横(h),折(z),撇(p),组合起来是：hzp
        self.stroke_dict = load_dict_file(stroke_path)
        self.proper_names_by_length: Dict[int, Set[str]] = defaultdict(set)
        self.char_pos_index: Dict[int, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.pinyin_pos_index: Dict[int, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.stroke_pos_index: Dict[int, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.pinyin_cache: Dict[str, List[str]] = {}
        self.stroke_cache: Dict[str, str] = {}
        for name in self.proper_names:
            self.proper_names_by_length[len(name)].add(name)
            for idx, char in enumerate(name):
                self.char_pos_index[len(name)][(idx, char)].add(name)
                self.pinyin_pos_index[len(name)][(idx, self.get_pinyin(char)[0])].add(name)
                stroke = self.get_stroke(char)
                if stroke:
                    self.stroke_pos_index[len(name)][(idx, stroke)].add(name)

    def get_stroke(self, char):
        """
        取笔画
        :param char:
        :return:
        """
        if char not in self.stroke_cache:
            self.stroke_cache[char] = self.stroke_dict.get(char, '')
        return self.stroke_cache[char]

    def get_pinyin(self, char):
        if char not in self.pinyin_cache:
            self.pinyin_cache[char] = pypinyin.lazy_pinyin(char)
        return self.pinyin_cache[char]

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
        if char1 == char2:
            return 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese_char(char1) != is_chinese_char(char2):
            return 0.0
        if not is_chinese_char(char1):
            return 0.0
        char_stroke1 = self.get_stroke(char1)
        char_stroke2 = self.get_stroke(char2)
        if not char_stroke1 or not char_stroke2:
            return 0.0
        # 相似度计算：1-编辑距离
        return 1.0 - edit_distance(char_stroke1, char_stroke2)

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
        if char1 == char2:
            return 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese_char(char1) != is_chinese_char(char2):
            return 0.0
        if not is_chinese_char(char1):
            return 0.0
        char_pinyin1 = self.get_pinyin(char1)[0]
        char_pinyin2 = self.get_pinyin(char2)[0]
        if not char_pinyin1 or not char_pinyin2:
            return 0.0
        # 相似度计算：1-编辑距离
        return 1.0 - edit_distance(char_pinyin1, char_pinyin2)

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

    def get_candidate_names(self, word):
        """
        按词长、位置索引召回候选专名，避免每个片段扫描全量专名词典。
        :param word: 待纠错片段
        :return: list, 候选专名集合
        """
        word_len = len(word)
        names = self.proper_names_by_length.get(word_len)
        if not names:
            return []
        if word_len == 1:
            return list(names)

        char_buckets = [self.char_pos_index[word_len].get((idx, char), set()) for idx, char in enumerate(word)]
        pinyin_buckets = [self.pinyin_pos_index[word_len].get((idx, self.get_pinyin(char)[0]), set()) for idx, char in enumerate(word)]
        stroke_buckets = [self.stroke_pos_index[word_len].get((idx, self.get_stroke(char)), set()) for idx, char in enumerate(word)]

        votes = defaultdict(int)
        for buckets in (char_buckets, pinyin_buckets, stroke_buckets):
            for skip_idx in range(word_len):
                selected = [bucket for idx, bucket in enumerate(buckets) if idx != skip_idx and bucket]
                if len(selected) != word_len - 1:
                    continue
                matched = set.intersection(*selected)
                for name in matched:
                    votes[name] += 1
        return sorted(votes, key=lambda name: votes[name], reverse=True)

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
        original_sentence = sentence  # 保存原始输入句子
        text_new = ''
        details = []
        # 切分为短句
        sentences = split_text_into_sentences_by_symbol(sentence, include_symbol=True)
        for short_sent, idx in sentences:
            current_sent = short_sent  # 当前处理的短句
            # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
            if cut_type == 'char':
                ngrams = [
                    (short_sent[i:i + word_len], i)
                    for word_len in range(min_word_length, max_word_length + 1)
                    for i in range(0, len(short_sent) - word_len + 1)
                ]
            else:
                sentence_words = segment(short_sent, cut_type=cut_type)
                ngram_items = NgramUtil.ngrams(sentence_words, ngram, join_string="_") or []
                ngrams = []
                for item in (i.replace("_", "") for i in ngram_items if i):
                    cur_idx = short_sent.find(item)
                    while cur_idx != -1:
                        ngrams.append((item, cur_idx))
                        cur_idx = short_sent.find(item, cur_idx + 1)
            # 去重并过滤词长度
            ngrams = list(dict.fromkeys((i, pos) for i, pos in ngrams if min_word_length <= len(i) <= max_word_length))
            
            # 收集所有需要纠错的信息，避免修改current_sent影响后续位置计算
            corrections = []
            for cur_item, cur_idx in ngrams:
                if cur_item in self.proper_names:
                    continue
                best_name = None
                best_score = sim_threshold
                for name in self.get_candidate_names(cur_item):
                    score = self.get_word_similarity_score(cur_item, name)
                    if score > best_score:
                        best_name = name
                        best_score = score
                if best_name and cur_item != best_name:
                    corrections.append((cur_item, best_name, cur_idx, best_score))
            
            # 优先保留长词、高分纠错，避免短片段与长专名重叠后重复替换
            filtered_corrections = []
            occupied = set()
            for cur_item, name, cur_idx, score in sorted(corrections, key=lambda x: (len(x[0]), x[3]), reverse=True):
                span = set(range(cur_idx, cur_idx + len(cur_item)))
                if occupied & span:
                    continue
                occupied.update(span)
                filtered_corrections.append((cur_item, name, cur_idx))
            # 按位置从后往前排序，避免前面的修改影响后面的位置
            filtered_corrections.sort(key=lambda x: x[2], reverse=True)
            
            # 应用所有纠错
            for cur_item, name, cur_idx in filtered_corrections:
                current_sent = current_sent[:cur_idx] + name + current_sent[(cur_idx + len(cur_item)):]
                details.append((cur_item, name, idx + cur_idx + start_idx))
            
            text_new += current_sent
        return {'source': original_sentence, 'target': text_new, 'errors': details}

    def correct_batch(self, sentences: List[str], **kwargs):
        """
        批量句子纠错
        :param sentences: 句子文本列表
        :param kwargs: 其他参数
        :return: list of {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        return [self.correct(s, **kwargs) for s in sentences]

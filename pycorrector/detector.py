# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: error word detector
import codecs
import kenlm
from pycorrector import config
import time

import numpy as np

from pycorrector.tokenizer import Tokenizer
from pycorrector.utils.logger import logger
from pycorrector.utils.text_utils import uniform, is_alphabet_string

PUNCTUATION_LIST = "。，,、？：；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-"
error_type = {"confusion": 1, "word": 2, "char": 3}


class Detector(object):
    def __init__(self, language_model_path=config.language_model_path,
                 word_freq_path=config.word_freq_path,
                 custom_word_freq_path=config.custom_word_freq_path,
                 custom_confusion_path=config.custom_confusion_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 stopwords_path=config.stopwords_path):
        self.name = 'detector'
        self.language_model_path =  language_model_path
        self.word_freq_path = word_freq_path
        self.custom_word_freq_path =custom_word_freq_path
        self.custom_confusion_path =  custom_confusion_path
        self.person_name_path = person_name_path
        self.place_name_path =place_name_path
        self.stopwords_path = stopwords_path
        self.is_char_error_detect = True
        self.is_word_error_detect = True
        self.initialized_detector = False

    def initialize_detector(self):
        t1 = time.time()
        self.lm = kenlm.Model(self.language_model_path)
        t2 = time.time()
        logger.debug(
            'Loaded language model: %s, spend: %s s' % (self.language_model_path, str(t2 - t1)))
        # 词、频数dict
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)
        t3 = time.time()
        logger.debug('Loaded word freq file: %s, size: %d, spend: %s s' %
                     (self.word_freq_path, len(self.word_freq), str(t3 - t2)))
        # 自定义混淆集
        self.custom_confusion = self._get_custom_confusion_dict(self.custom_confusion_path)
        t4 = time.time()
        logger.debug('Loaded confusion file: %s, size: %d, spend: %s s' %
                     (self.custom_confusion_path, len(self.custom_confusion), str(t4 - t3)))
        # 自定义切词词典
        self.custom_word_freq = self.load_word_freq_dict(self.custom_word_freq_path)
        self.person_names = self.load_word_freq_dict(self.person_name_path)
        self.place_names = self.load_word_freq_dict(self.place_name_path)
        self.stopwords = self.load_word_freq_dict(self.stopwords_path)
        # 合并切词词典及自定义词典
        self.custom_word_freq.update(self.person_names)
        self.custom_word_freq.update(self.place_names)
        self.custom_word_freq.update(self.stopwords)

        self.word_freq.update(self.custom_word_freq)
        t5 = time.time()
        logger.debug('Loaded custom word file: %s, size: %d, spend: %s s' %
                     (self.custom_confusion_path, len(self.custom_word_freq), str(t5 - t4)))
        logger.debug('Loaded all word freq file done, size: %d' % len(self.word_freq))
        self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
                                   custom_confusion_dict=self.custom_confusion)
        t6 = time.time()
        logger.info('Loaded dict ok, spend: %s s' % str(t6 - t1))
        self.initialized_detector = True

    def check_detector_initialized(self):
        if not self.initialized_detector:
            self.initialize_detector()

    @staticmethod
    def load_word_freq_dict(path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = {}
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                info = line.split()
                if len(info) < 1:
                    continue
                word = info[0]
                # 取词频，默认1
                freq = int(info[1]) if len(info) > 1 else 1
                word_freq[word] = freq
        return word_freq

    def _get_custom_confusion_dict(self, path):
        """
        取自定义困惑集
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        confusion = {}
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                info = line.split()
                if len(info) < 2:
                    continue
                variant = info[0]
                origin = info[1]
                freq = int(info[2]) if len(info) > 2 else 1
                self.word_freq[origin] = freq
                confusion[variant] = origin
        return confusion

    def set_language_model_path(self, path):
        self.check_detector_initialized()
        self.lm = kenlm.Model(path)
        logger.info('Loaded language model: %s' % path)

    def set_custom_confusion_dict(self, path):
        self.check_detector_initialized()
        custom_confusion = self._get_custom_confusion_dict(path)
        self.custom_confusion.update(custom_confusion)
        logger.info('Loaded confusion path: %s, size: %d' % (path, len(custom_confusion)))

    def set_custom_word(self, path):
        self.check_detector_initialized()
        word_freqs = self.load_word_freq_dict(path)
        # 合并字典
        self.custom_word_freq.update(word_freqs)
        # 合并切词词典及自定义词典
        self.word_freq.update(self.custom_word_freq)
        self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
                                   custom_confusion_dict=self.custom_confusion)
        for k, v in word_freqs.items():
            self.set_word_frequency(k, v)
        logger.info('Loaded custom word path: %s, size: %d' % (path, len(word_freqs)))

    def enable_char_error(self, enable=True):
        """
        is open char error detect
        :param enable:
        :return:
        """
        self.is_char_error_detect = enable

    def enable_word_error(self, enable=True):
        """
        is open word error detect
        :param enable:
        :return:
        """
        self.is_word_error_detect = enable

    def ngram_score(self, chars):
        """
        取n元文法得分
        :param chars: list, 以词或字切分
        :return:
        """
        self.check_detector_initialized()
        return self.lm.score(' '.join(chars), bos=False, eos=False)

    def ppl_score(self, words):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param words: list, 以词或字切分
        :return:
        """
        self.check_detector_initialized()
        return self.lm.perplexity(' '.join(words))

    def word_frequency(self, word):
        """
        取词在样本中的词频
        :param word:
        :return:
        """
        self.check_detector_initialized()
        return self.word_freq.get(word, 0)

    def set_word_frequency(self, word, num):
        """
        更新在样本中的词频
        """
        self.check_detector_initialized()
        self.word_freq[word] = num
        return self.word_freq

    @staticmethod
    def _check_contain_error(maybe_err, maybe_errors):
        """
        检测错误集合(maybe_errors)是否已经包含该错误位置（maybe_err)
        :param maybe_err: [error_word, begin_pos, end_pos, error_type]
        :param maybe_errors:
        :return:
        """
        error_word_idx = 0
        begin_idx = 1
        end_idx = 2
        for err in maybe_errors:
            if maybe_err[error_word_idx] in err[error_word_idx] and maybe_err[begin_idx] >= err[begin_idx] and \
                            maybe_err[end_idx] <= err[end_idx]:
                return True
        return False

    def _add_maybe_error_item(self, maybe_err, maybe_errors):
        """
        新增错误
        :param maybe_err:
        :param maybe_errors:
        :return:
        """
        if maybe_err not in maybe_errors and not self._check_contain_error(maybe_err, maybe_errors):
            maybe_errors.append(maybe_err)

    @staticmethod
    def _get_maybe_error_index(scores, ratio=0.6745, threshold=1.4):
        """
        取疑似错字的位置，通过平均绝对离差（MAD）
        :param scores: np.array
        :param threshold: 阈值越小，得到疑似错别字越多
        :return:
        """
        scores = np.array(scores)
        if len(scores.shape) == 1:
            scores = scores[:, None]
        median = np.median(scores, axis=0)  # get median of all scores
        margin_median = np.sqrt(np.sum((scores - median) ** 2, axis=-1))  # deviation from the median
        # 平均绝对离差值
        med_abs_deviation = np.median(margin_median)
        if med_abs_deviation == 0:
            return []
        y_score = ratio * margin_median / med_abs_deviation
        # 打平
        scores = scores.flatten()
        maybe_error_indices = np.where((y_score > threshold) & (scores < median))
        # 取全部疑似错误字的index
        return list(maybe_error_indices[0])

    def detect(self, sentence):
        """
        检测句子中的疑似错误信息，包括[词、位置、错误类型]
        :param sentence:
        :return: [error_word, begin_pos, end_pos, error_type]
        """
        maybe_errors = []
        if not sentence.strip():
            return maybe_errors
        self.check_detector_initialized()
        # 文本归一化
        sentence = uniform(sentence)
        # 切词
        tokens = self.tokenizer.tokenize(sentence)
        # print(tokens)
        # 自定义混淆集加入疑似错误词典
        for confuse in self.custom_confusion:
            idx = sentence.find(confuse)
            if idx > -1:
                maybe_err = [confuse, idx, idx + len(confuse), error_type["confusion"]]
                self._add_maybe_error_item(maybe_err, maybe_errors)

        if self.is_word_error_detect:
            # 未登录词加入疑似错误词典
            for word, begin_idx, end_idx in tokens:
                # pass blank
                if not word.strip():
                    continue
                # punctuation
                if word in PUNCTUATION_LIST:
                    continue
                # pass num
                if word.isdigit():
                    continue
                # pass alpha
                if is_alphabet_string(word.lower()):
                    continue
                # in dict
                if word in self.word_freq:
                    continue
                maybe_err = [word, begin_idx, end_idx, error_type["word"]]
                self._add_maybe_error_item(maybe_err, maybe_errors)

        if self.is_char_error_detect:
            # 语言模型检测疑似错误字
            ngram_avg_scores = []
            try:
                for n in [2, 3]:
                    scores = []
                    for i in range(len(sentence) - n + 1):
                        word = sentence[i:i + n]
                        score = self.ngram_score(list(word))
                        scores.append(score)
                    if not scores:
                        continue
                    # 移动窗口补全得分
                    for _ in range(n - 1):
                        scores.insert(0, scores[0])
                        scores.append(scores[-1])
                    avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
                    ngram_avg_scores.append(avg_scores)

                # 取拼接后的ngram平均得分
                sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
                # 取疑似错字信息
                for i in self._get_maybe_error_index(sent_scores):
                    maybe_err = [sentence[i], i, i + 1, error_type["char"]]
                    self._add_maybe_error_item(maybe_err, maybe_errors)
            except IndexError as ie:
                logger.warn("index error, sentence:" + sentence + str(ie))
            except Exception as e:
                logger.warn("detect error, sentence:" + sentence + str(e))
        return sorted(maybe_errors, key=lambda k: k[1], reverse=False)

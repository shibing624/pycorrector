# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: error word detector

import codecs
import os
import re
import time

import numpy as np

from . import config
from .utils.get_file import get_file
from .utils.logger import logger
from .utils.text_utils import uniform, is_alphabet_string, convert_to_unicode, is_chinese_string
from .utils.tokenizer import Tokenizer

# \u4E00-\u9FA5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
re_han = re.compile("([\u4E00-\u9Fa5a-zA-Z0-9+#&]+)", re.U)
re_skip = re.compile("(\r\n\\s)", re.U)


class ErrorType(object):
    # error_type = {"confusion": 1, "word": 2, "char": 3}
    confusion = 'confusion'
    word = 'word'
    char = 'char'


class Detector(object):
    pre_trained_language_models = {
        # 语言模型 2.95GB
        'zh_giga.no_cna_cmn.prune01244.klm': 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        # 人民日报训练语言模型 20MB
        'people_chars_lm.klm': 'https://www.borntowin.cn/mm/emb_models/people_chars_lm.klm'
    }

    def __init__(self,
                 language_model_path=config.language_model_path,
                 word_freq_path=config.word_freq_path,
                 custom_word_freq_path='',
                 custom_confusion_path='',
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 stopwords_path=config.stopwords_path
                 ):
        self.name = 'detector'
        self.language_model_path = language_model_path
        self.word_freq_path = word_freq_path
        self.custom_word_freq_path = custom_word_freq_path
        self.custom_confusion_path = custom_confusion_path
        self.person_name_path = person_name_path
        self.place_name_path = place_name_path
        self.stopwords_path = stopwords_path
        self.is_char_error_detect = True
        self.is_word_error_detect = True
        self.initialized_detector = False
        self.lm = None
        self.word_freq = None
        self.custom_confusion = None
        self.custom_word_freq = None
        self.person_names = None
        self.place_names = None
        self.stopwords = None
        self.tokenizer = None

    def _initialize_detector(self):
        t1 = time.time()
        try:
            import kenlm
        except ImportError:
            raise ImportError('pycorrector dependencies are not fully installed, '
                              'they are required for statistical language model.'
                              'Please use "pip install kenlm" to install it.'
                              'if you are Win, Please install kenlm in cgwin.')
        if not os.path.exists(self.language_model_path):
            filename = self.pre_trained_language_models.get(self.language_model_path,
                                                            'zh_giga.no_cna_cmn.prune01244.klm')
            url = self.pre_trained_language_models.get(filename)
            get_file(
                filename, url, extract=True,
                cache_dir='~',
                cache_subdir=config.USER_DATA_DIR,
                verbose=1
            )
        self.lm = kenlm.Model(self.language_model_path)
        t2 = time.time()
        logger.debug('Loaded language model: %s, spend: %.3f s.' % (self.language_model_path, t2 - t1))

        # 词、频数dict
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)
        # 自定义混淆集
        self.custom_confusion = self._get_custom_confusion_dict(self.custom_confusion_path)
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
        self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
                                   custom_confusion_dict=self.custom_confusion)
        t3 = time.time()
        logger.debug('Loaded dict file, spend: %.3f s.' % (t3 - t2))
        self.initialized_detector = True

    def check_detector_initialized(self):
        if not self.initialized_detector:
            self._initialize_detector()

    @staticmethod
    def load_word_freq_dict(path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = {}
        if path:
            if not os.path.exists(path):
                logger.warning('file not found.%s' % path)
                return word_freq
            else:
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
        if path:
            if not os.path.exists(path):
                logger.warning('file not found.%s' % path)
                return confusion
            else:
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
        import kenlm
        self.lm = kenlm.Model(path)
        logger.debug('Loaded language model: %s' % path)

    def set_custom_confusion_dict(self, path):
        self.check_detector_initialized()
        self.custom_confusion = self._get_custom_confusion_dict(path)
        logger.debug('Loaded confusion path: %s, size: %d' % (path, len(self.custom_confusion)))

    def set_custom_word_freq(self, path):
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
        logger.debug('Loaded custom word path: %s, size: %d' % (path, len(word_freqs)))

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
        :return: dict
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
        :param maybe_errors:list
        :return: bool
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
    def _get_maybe_error_index(scores, ratio=0.6745, threshold=2):
        """
        取疑似错字的位置，通过平均绝对离差（MAD）
        :param scores: np.array
        :param ratio: 正态分布表参数
        :param threshold: 阈值越小，得到疑似错别字越多
        :return: 全部疑似错误字的index: list
        """
        result = []
        scores = np.array(scores)
        if len(scores.shape) == 1:
            scores = scores[:, None]
        median = np.median(scores, axis=0)  # get median of all scores
        margin_median = np.abs(scores - median).flatten()  # deviation from the median
        # 平均绝对离差值
        med_abs_deviation = np.median(margin_median)
        if med_abs_deviation == 0:
            return result
        y_score = ratio * margin_median / med_abs_deviation
        # 打平
        scores = scores.flatten()
        maybe_error_indices = np.where((y_score > threshold) & (scores < median))
        # 取全部疑似错误字的index
        result = list(maybe_error_indices[0])
        return result

    @staticmethod
    def _get_maybe_error_index_by_stddev(scores, n=2):
        """
        取疑似错字的位置，通过平均值上下n倍标准差之间属于正常点
        :param scores: list, float
        :param n: n倍
        :return: 全部疑似错误字的index: list
        """
        std = np.std(scores, ddof=1)
        mean = np.mean(scores)
        down_limit = mean - n * std
        upper_limit = mean + n * std
        maybe_error_indices = np.where((scores > upper_limit) | (scores < down_limit))
        # 取全部疑似错误字的index
        result = list(maybe_error_indices[0])
        return result

    @staticmethod
    def is_filter_token(token):
        """
        是否为需过滤字词
        :param token: 字词
        :return: bool
        """
        result = False
        # pass blank
        if not token.strip():
            result = True
        # pass num
        if token.isdigit():
            result = True
        # pass alpha
        if is_alphabet_string(token.lower()):
            result = True
        # pass not chinese
        if not is_chinese_string(token):
            result = True
        return result

    @staticmethod
    def split_2_short_text(text, include_symbol=False):
        """
        长句切分为短句
        :param text: str
        :param include_symbol: bool
        :return: (sentence, idx)
        """
        result = []
        blocks = re_han.split(text)
        start_idx = 0
        for blk in blocks:
            if not blk:
                continue
            if include_symbol:
                result.append((blk, start_idx))
            else:
                if re_han.match(blk):
                    result.append((blk, start_idx))
            start_idx += len(blk)
        return result

    @staticmethod
    def split_text_by_maxlen(text, maxlen=512):
        """
        长句切分为短句，每个短句maxlen个字
        :param text: str
        :param maxlen: int, 最大长度
        :return: list, (sentence, idx)
        """
        result = []
        for i in range(0, len(text), maxlen):
            result.append((text[i:i + maxlen], i))
        return result

    def detect(self, text):
        """
        文本错误检测
        :param text: 长文本
        :return: 错误index
        """
        maybe_errors = []
        if not text.strip():
            return maybe_errors
        # 初始化
        self.check_detector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 文本归一化
        text = uniform(text)
        # 长句切分为短句
        blocks = self.split_2_short_text(text)
        for blk, idx in blocks:
            maybe_errors += self.detect_short(blk, idx)
        return maybe_errors

    def detect_short(self, sentence, start_idx=0):
        """
        检测句子中的疑似错误信息，包括[词、位置、错误类型]
        :param sentence:
        :param start_idx:
        :return: list[list], [error_word, begin_pos, end_pos, error_type]
        """
        maybe_errors = []
        # 初始化
        self.check_detector_initialized()
        # 自定义混淆集加入疑似错误词典
        for confuse in self.custom_confusion:
            idx = sentence.find(confuse)
            if idx > -1:
                maybe_err = [confuse, idx + start_idx, idx + len(confuse) + start_idx, ErrorType.confusion]
                self._add_maybe_error_item(maybe_err, maybe_errors)

        if self.is_word_error_detect:
            # 切词
            tokens = self.tokenizer.tokenize(sentence)
            # 未登录词加入疑似错误词典
            for token, begin_idx, end_idx in tokens:
                # pass filter word
                if self.is_filter_token(token):
                    continue
                # pass in dict
                if token in self.word_freq:
                    continue
                maybe_err = [token, begin_idx + start_idx, end_idx + start_idx, ErrorType.word]
                self._add_maybe_error_item(maybe_err, maybe_errors)

        if self.is_char_error_detect:
            # 语言模型检测疑似错误字
            try:
                ngram_avg_scores = []
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

                if ngram_avg_scores:
                    # 取拼接后的n-gram平均得分
                    sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
                    # 取疑似错字信息
                    for i in self._get_maybe_error_index(sent_scores):
                        token = sentence[i]
                        # pass filter word
                        if self.is_filter_token(token):
                            continue
                        # pass in stop word dict
                        if token in self.stopwords:
                            continue
                        # token, begin_idx, end_idx, error_type
                        maybe_err = [token, i + start_idx, i + start_idx + 1,
                                     ErrorType.char]
                        self._add_maybe_error_item(maybe_err, maybe_errors)
            except IndexError as ie:
                logger.warn("index error, sentence:" + sentence + str(ie))
            except Exception as e:
                logger.warn("detect error, sentence:" + sentence + str(e))
        return sorted(maybe_errors, key=lambda k: k[1], reverse=False)

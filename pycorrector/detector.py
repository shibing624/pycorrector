# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: error word detector
import codecs
import kenlm
import os

import numpy as np

import pycorrector.config as config
from pycorrector.utils.io_utils import dump_pkl
from pycorrector.utils.io_utils import get_logger
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.text_utils import uniform, tokenize

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_logger = get_logger(__file__)
# trigram_word_path = os.path.join(pwd_path, 'data/kenlm/people_words.klm')
# trigram_word = kenlm.Model(trigram_word_path)
# print('Loaded trigram_word language model from {}'.format(trigram_word_path))

trigram_char_path = os.path.join(pwd_path, config.language_model_path)
trigram_char = kenlm.Model(trigram_char_path)
default_logger.debug('Loaded trigram_word language model from {}'.format(trigram_char_path))

PUNCTUATION_LIST = "。，,、？：；{}[]【】“‘’”《》/！%……（）<>@#$~^￥%&*\"\'=+-"


def load_word_freq_dict(path):
    word_freq = {}
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            info = line.split()
            word = info[0]
            freq = int(info[1])
            word_freq[word] = freq
    return word_freq


# 字频统计
word_freq_path = os.path.join(pwd_path, config.word_freq_path)
word_freq_model_path = os.path.join(pwd_path, config.word_freq_model_path)
if os.path.exists(word_freq_model_path):
    word_freq = load_pkl(word_freq_model_path)
else:
    default_logger.debug('load word freq from text file:', word_freq_path)
    word_freq = load_word_freq_dict(word_freq_path)
    dump_pkl(word_freq, word_freq_model_path)


def get_ngram_score(chars, mode=trigram_char):
    """
    取n元文法得分
    :param chars: list, 以词或字切分
    :param mode:
    :return:
    """
    return mode.score(' '.join(chars), bos=False, eos=False)


def get_ppl_score(words, mode=trigram_char):
    """
    取语言模型困惑度得分，越小句子越通顺
    :param words: list, 以词或字切分
    :param mode:
    :return:
    """
    return mode.perplexity(' '.join(words))


def get_frequency(word):
    """
    取词在样本中的词频
    :param word:
    :return:
    """
    return word_freq.get(word, 0)


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
    y_score = ratio * margin_median / med_abs_deviation
    # 打平
    scores = scores.flatten()
    maybe_error_indices = np.where((y_score > threshold) & (scores < median))
    # 取全部疑似错误字的index
    return list(maybe_error_indices[0])


def detect(sentence):
    maybe_error_indices = set()
    # 文本归一化
    sentence = uniform(sentence)
    # 切词
    tokens = tokenize(sentence)
    # 未登录词加入疑似错误字典
    for word, begin_idx, end_idx in tokens:
        if word not in PUNCTUATION_LIST and word not in word_freq.keys():
            for i in range(begin_idx, end_idx):
                maybe_error_indices.add(i)
    # 语言模型检测疑似错字
    ngram_avg_scores = []
    try:
        for n in [2, 3]:
            scores = []
            for i in range(len(sentence) - n + 1):
                word = sentence[i:i + n]
                score = get_ngram_score(list(word), mode=trigram_char)
                scores.append(score)
            # 移动窗口补全得分
            for _ in range(n - 1):
                scores.insert(0, scores[0])
                scores.append(scores[-1])
            avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
            ngram_avg_scores.append(avg_scores)

        # 取拼接后的ngram平均得分
        sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
        maybe_error_char_indices = _get_maybe_error_index(sent_scores)
        # 合并字、词错误
        maybe_error_indices |= set(maybe_error_char_indices)
    except IndexError:
        print("index error, sentence:", sentence)
        pass
    except:
        print("detect error, sentence:", sentence)
    return sorted(maybe_error_indices)


if __name__ == '__main__':
    sent = '少先队员因该为老人让坐'
    # sent = '机七学习是人工智能领遇最能体现智能的一个分知'
    error_list = detect(sent)
    print(error_list)

    sent_chars = [sent[i] for i in error_list]
    print(sent_chars)

    from utils.text_utils import segment, tokenize

    print(get_ngram_score(segment(sent)))
    print(get_ppl_score(segment(sent)))

    print(get_ngram_score(list(sent), mode=trigram_char))
    print(get_ppl_score(list(sent), mode=trigram_char))

    sent = '少先队员应该为老人让座'
    print(detect(sent))
    print(get_ngram_score(segment(sent)))
    print(get_ppl_score(segment(sent)))

    print(get_ngram_score(list(sent), mode=trigram_char))
    print(get_ppl_score(list(sent), mode=trigram_char))

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: error word detector
import codecs
import kenlm
import os
import pdb
import sys
sys.path.append("../")

import numpy as np
import jieba.posseg as pseg

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
word_dict_path = os.path.join(pwd_path, config.word_dict_path)
word_dict_model_path = os.path.join(pwd_path, config.word_dict_model_path)
if os.path.exists(word_dict_model_path):
    word_freq = load_pkl(word_dict_model_path)
else:
    default_logger.debug('load word freq from text file:', word_dict_path)
    word_freq = load_word_freq_dict(word_dict_path)
    dump_pkl(word_freq, word_dict_model_path)


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


def _get_maybe_error_index(scores, ratio=0.6745, threshold=0.5):
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

    # ######################
    # print(y_score)
    # pdb.set_trace()
    ######################

    # 取全部疑似错误字的index
    return list(maybe_error_indices[0])


def detect(sentence):
    maybe_error_indices = set()

    sentence = uniform(sentence)

    tokens = tokenize(sentence)

    # unknown chars
    for word, begin_idx, end_idx in tokens:
        if word not in PUNCTUATION_LIST and word not in word_freq.keys():
            for i in range(begin_idx, end_idx):
                maybe_error_indices.add(i)
                

    ngram_avg_scores = []
    try:
        for n in [1, 2, 3]:
            scores = []
            for i in range(len(sentence) - n + 1):
                word = sentence[i:i + n]
                score = get_ngram_score(list(word), mode=trigram_char)
                scores.append(score)

            for _ in range(n - 1):
                scores.insert(0, scores[0])
                scores.append(scores[-1])

            avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
            ngram_avg_scores.append(avg_scores)


        sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
        maybe_error_char_indices = _get_maybe_error_index(sent_scores)




        maybe_error_indices |= set(maybe_error_char_indices)
    except IndexError as ie:
        print("index error, sentence:", sentence, ie)
        pass
    except Exception as e:
        print("detect error, sentence:", sentence, e)

    # # to get rid of special nouns like name
    seg = pseg.lcut(sentence)
    # # in the form of list of pair(w.word, w.flag)
    word = [w.word for w in seg]
    tag  = [w.flag for w in seg]

    for i in range(len(tag)):
        if tag[i] in {'nz','nr','nt','ns'}:
            if len(word[i]) > 1:
                maybe_error_indices -= set(range(len(''.join(word[:i])), \
                                                 len(''.join(word[:i + 1]))))
            elif i + 1 < len(tag) and tag[i + 1] in {'nz','nr','nt','ns'}:
                maybe_error_indices -= set(range(len(''.join(word[:i])), \
                                                 len(''.join(word[:i + 2]))))               


    # #####################
    # print(maybe_error_indices)
    # print([sentence[i] for i in maybe_error_indices])
    # pdb.set_trace()
    # #####################

    return sorted(maybe_error_indices)



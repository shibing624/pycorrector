# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 配置切词器
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import jieba
from jieba import posseg

jieba.setLogLevel(log_level="ERROR")


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


class Tokenizer(object):
    def __init__(self, dict_path='', custom_word_freq_dict=None, custom_confusion_dict=None):
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

        # 加载混淆集词典
        if custom_confusion_dict:
            for k, word in custom_confusion_dict.items():
                # 添加到分词器的自定义词典中
                self.model.add_word(k)
                self.model.add_word(word)

    def tokenize(self, unicode_sentence, mode="search"):
        """
        切词并返回切词位置, search mode用于错误扩召回
        :param unicode_sentence: query
        :param mode: search, default, ngram
        :param HMM: enable HMM
        :return: (w, start, start + width) model='default'
        """
        if mode == 'ngram':
            n = 2
            result_set = set()
            tokens = self.model.lcut(unicode_sentence)
            tokens_len = len(tokens)
            start = 0
            for i in range(0, tokens_len):
                w = tokens[i]
                width = len(w)
                result_set.add((w, start, start + width))
                for j in range(i, i + n):
                    gram = "".join(tokens[i:j + 1])
                    gram_width = len(gram)
                    if i + j > tokens_len:
                        break
                    result_set.add((gram, start, start + gram_width))
                start += width
            results = list(result_set)
            result = sorted(results, key=lambda x: x[-1])
        else:
            result = list(self.model.tokenize(unicode_sentence, mode=mode))
        return result


if __name__ == '__main__':
    text = "这个消息在北京城里不胫儿走"
    print(text)
    t = Tokenizer()
    print('deault', t.tokenize(text, 'default'))
    print('search', t.tokenize(text, 'search'))
    print('ngram', t.tokenize(text, 'ngram'))

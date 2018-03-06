# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import re
from codecs import open
from .cn_spell import correct
from .util import load_pkl


def get_bcmi_corpus(line, left_symbol='（（', right_symbol='））'):
    error_sentence, correct_sentence = '', ''
    left_words, correct_words, right_words = [], [], []
    if left_symbol in line and right_symbol in line:
        left_pattern = re.compile('(\w+)' + left_symbol)
        correct_pattern = re.compile('(\w+)' + right_symbol)
        right_pattern = re.compile(right_symbol + '(\w+)')
        left_words = left_pattern.findall(line)
        correct_words = correct_pattern.findall(line)
        right_words = right_pattern.findall(line)
    if left_words and right_words and correct_words:
        for left, right, chr in zip(left_words, right_words, correct_words):
            error_sent = left + right
            error_sentence += error_sent
            new_l = left.replace(left[-len(chr)], chr, 1)
            correct_sent = new_l + right
            correct_sentence += correct_sent
    return error_sentence, correct_sentence


def eval_bcmi_data(data_path, verbose=False):
    sentence_size = 1
    right_count = 0
    right_result = dict()
    wrong_result = dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            error_sentence, right_sentence = get_bcmi_corpus(line)
            if not error_sentence:
                continue
            pred_sentence, pred_detail = correct(error_sentence, True)
            if verbose:
                print('input sentence:', error_sentence)
                print('pred sentence:', pred_sentence)
                print('right sentence:', right_sentence)
            sentence_size += 1
            if right_sentence == pred_sentence:
                right_count += 1
                right_result[error_sentence] = [right_sentence, pred_sentence]
            else:
                wrong_result[error_sentence] = [right_sentence, pred_sentence]
    if verbose:
        print('right count:', right_count, ';sentence size:', sentence_size)
    return right_count / sentence_size, right_result, wrong_result


def eval_sighan_corpus(pkl_path, verbose=False):
    sighan_data = load_pkl(pkl_path)
    total_count = 1
    right_count = 0
    right_result = dict()
    wrong_result = dict()
    for error_sentence, right_detail in sighan_data:
        pred_sentence, pred_detail = correct(error_sentence, True)
        if verbose:
            print('input sentence:', error_sentence)
            print('pred sentence:', pred_sentence)
        for (right_loc, right_w, right_r), (pred_loc, pred_w, pred_r) in zip(right_detail, pred_detail):
            total_count += 1
            if right_r == pred_r:
                right_count += 1
                right_result[error_sentence] = [right_r, pred_r]
            else:
                wrong_result[error_sentence] = [right_r, pred_r]
            if verbose:
                print('right: {} => {} , index: {}'.format(right_w, right_r, right_loc))
    if verbose:
        print('right count:', right_count, ';total count:', total_count)
    return right_count / total_count, right_result, wrong_result

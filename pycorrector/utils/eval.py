# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from codecs import open

from pycorrector import correct
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.math_utils import find_all_idx


def get_bcmi_corpus(line, left_symbol='（（', right_symbol='））'):
    """
    转换原始文本为encoder-decoder列表
    :param line: 王老师心（（性））格温和，态度和爱（（蔼）），教学有方，得到了许多人的好平（（评））。
    :param left_symbol:
    :param right_symbol:
    :return: ["王老师心格温和，态度和爱，教学有方，得到了许多人的好平。" , "王老师性格温和，态度和蔼，教学有方，得到了许多人的好评。"]
    """
    error_sentence = ''
    correct_sentence = ''
    detail = []
    if left_symbol not in line or right_symbol not in line:
        return error_sentence, correct_sentence, detail

    left_ids = find_all_idx(line, left_symbol)
    right_ids = find_all_idx(line, right_symbol)
    if len(left_ids) != len(right_ids):
        return error_sentence, correct_sentence, detail
    begin = 0
    for left, right in zip(left_ids, right_ids):
        correct_len = right - left - len(left_symbol)
        correct_word = line[(left + len(left_symbol)):right]
        error_sentence += line[begin:left]
        correct_sentence += line[begin:(left - correct_len)] + correct_word
        begin = right + len(right_symbol)
        detail.append(correct_word)
    error_sentence += line[begin:]
    correct_sentence += line[begin:]
    return error_sentence, correct_sentence, detail


def eval_bcmi_data(data_path, verbose=False):
    sentence_size = 1
    right_count = 0
    right_result = dict()
    wrong_result = dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            error_sentence, right_sentence, right_detail = get_bcmi_corpus(line)
            if not error_sentence:
                continue
            pred_sentence, pred_detail = correct(error_sentence)
            if verbose:
                print('input sentence:', error_sentence)
                print('pred sentence:', pred_sentence, pred_detail)
                print('right sentence:', right_sentence, right_detail)
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
    for error_sentence, right_detail in sighan_data:
        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        pred_sentence, pred_detail = correct(error_sentence)
        if verbose:
            print('input sentence:', error_sentence, right_detail)
            print('pred sentence:', pred_sentence, pred_detail)
        if len(right_detail) != len(pred_detail):
            total_count += 1
        else:
            right_count += 1
    return right_count / total_count


if __name__ == "__main__":
    lst = ['少先队员因（（应））该为老人让坐（（座））。',
           '王老师心（（性））格温和，态度和爱（（蔼）），教学有方，得到了许多人的好平（（评））。',
           '青蛙是庄家的好朋友，我们要宝（（保））护它们。']
    for i in lst:
        print(get_bcmi_corpus(i))

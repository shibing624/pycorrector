# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
import sys
import time
from codecs import open
from random import sample
from xml.dom import minidom

sys.path.append("../..")
import pycorrector
from pycorrector.utils.io_utils import load_json, save_json
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.math_utils import find_all_idx

pwd_path = os.path.abspath(os.path.dirname(__file__))
eval_data_path = os.path.join(pwd_path, '../data/eval_corpus.json')
sighan_2015_path = os.path.join(pwd_path, '../data/cn/sighan_2015/test.tsv')


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
    details = []
    if left_symbol not in line or right_symbol not in line:
        return error_sentence, correct_sentence, details

    left_ids = find_all_idx(line, left_symbol)
    right_ids = find_all_idx(line, right_symbol)
    if len(left_ids) != len(right_ids):
        return error_sentence, correct_sentence, details
    begin = 0
    for left, right in zip(left_ids, right_ids):
        correct_len = right - left - len(left_symbol)
        correct_word = line[(left + len(left_symbol)):right]
        error_sentence += line[begin:left]
        correct_sentence += line[begin:(left - correct_len)] + correct_word
        begin = right + len(right_symbol)
        details.append(correct_word)
    error_sentence += line[begin:]
    correct_sentence += line[begin:]
    n_details = []
    for i in details:
        idx = correct_sentence.find(i)
        end_idx = idx + len(i)
        error_item = error_sentence[idx:end_idx]
        n_details.append([error_item, i, idx, end_idx])
    return error_sentence, correct_sentence, n_details


def build_bcmi_corpus(data_path, output_path):
    corpus = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            error_sentence, correct_sentence, details = get_bcmi_corpus(line)
            if not error_sentence:
                continue
            line_dict = {"text": error_sentence, "correction": correct_sentence, "errors": details}
            corpus.append(line_dict)
        save_json(corpus, output_path)


def eval_bcmi_data(data_path, verbose=False):
    total_count = 0
    right_count = 0
    right_result = dict()
    wrong_result = dict()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            error_sentence, right_sentence, right_detail = get_bcmi_corpus(line)
            if not error_sentence:
                continue
            pred_sentence, pred_detail = pycorrector.correct(error_sentence)
            total_count += 1
            if right_sentence == pred_sentence:
                right_count += 1
                right_result[error_sentence] = [right_sentence, pred_sentence]
            else:
                wrong_result[error_sentence] = [right_sentence, pred_sentence]
                if verbose:
                    print('input sentence:', error_sentence)
                    print('pred sentence:', pred_sentence, pred_detail)
                    print('right sentence:', right_sentence, right_detail)
    if verbose:
        print('right count:', right_count, ';total_count:', total_count)
    right_rate = 0.0
    if total_count > 0:
        right_rate = right_count / total_count
    return right_rate, right_result, wrong_result


def build_sighan_corpus(data_path, output_path):
    corpus = []
    sighan_data = load_pkl(data_path)
    for error_sentence, error_details in sighan_data:
        ids = []
        error_word = ''
        right_word = ''
        if not error_details:
            continue
        for detail in error_details:
            idx = detail[0]
            error_word = detail[1]
            right_word = detail[2]
            begin_idx = idx - 1
            ids.append(begin_idx)
        correct_sentence = error_sentence.replace(error_word, right_word)
        details = []
        for i in ids:
            details.append([error_sentence[i], correct_sentence[i], i, i + 1])
        line_dict = {"text": error_sentence, "correction": correct_sentence, "errors": details}
        corpus.append(line_dict)
    save_json(corpus, output_path)


def build_cged_no_error_corpus(data_path, output_path, limit_size=500):
    corpus = []
    print('Parse data from %s' % data_path)
    dom_tree = minidom.parse(data_path)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    count = 0
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('TEXT')[0]. \
            childNodes[0].data.strip()
        # Input the correct text
        correction = doc.getElementsByTagName('CORRECTION')[0]. \
            childNodes[0].data.strip()

        if correction:
            count += 1
            line_dict = {"text": correction, "correction": correction, "errors": []}
            corpus.append(line_dict)
            if count > limit_size:
                break
    save_json(corpus, output_path)


def build_eval_corpus(output_eval_path=eval_data_path):
    """
    生成评估样本集，抽样分布可修改
    当前已经生成评估集，可以修改代码生成自己的样本分布
    :param output_eval_path:
    :return: json file
    """
    bcmi_path = os.path.join(pwd_path, '../data/cn/bcmi.txt')
    clp_path = os.path.join(pwd_path, '../data/cn/clp14_C1.pkl')
    sighan_path = os.path.join(pwd_path, '../data/cn/sighan15_A2.pkl')
    cged_path = os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml')

    char_error_path = os.path.join(pwd_path, './bcmi_corpus.json')
    build_bcmi_corpus(bcmi_path, char_error_path)
    char_errors = load_json(char_error_path)

    word_error_path = os.path.join(pwd_path, './sighan_corpus.json')
    build_sighan_corpus(sighan_path, word_error_path)
    word_errors = load_json(word_error_path)

    grammar_error_path = os.path.join(pwd_path, './clp_corpus.json')
    build_sighan_corpus(clp_path, grammar_error_path)
    grammar_errors = load_json(grammar_error_path)

    no_error_path = os.path.join(pwd_path, './noerror_corpus.json')
    build_cged_no_error_corpus(cged_path, no_error_path)
    no_errors = load_json(no_error_path)

    corpus = sample(char_errors, 100) + sample(word_errors, 100) + sample(grammar_errors, 100) + sample(no_errors, 200)
    save_json(corpus, output_eval_path)
    print("save eval corpus done", output_eval_path)
    os.remove(char_error_path)
    os.remove(word_error_path)
    os.remove(grammar_error_path)
    os.remove(no_error_path)


def eval_corpus500_by_model(correct_fn, input_eval_path=eval_data_path, verbose=True):
    """
    句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    corpus = load_json(input_eval_path)
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    for data_dict in corpus:
        src = data_dict.get('text', '')
        tgt = data_dict.get('correction', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        tgt_pred, pred_detail = correct_fn(src)
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt, errors)
            print('predict:', tgt_pred, pred_detail)

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                print('right')
            # 预测为正
            else:
                FP += 1
                print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                print('right')
            # 预测为负
            else:
                FN += 1
                print('wrong')
        total_num += 1
    spend_time = time.time() - start_time
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, cost time:{spend_time:.2f} s')
    return acc, precision, recall, f1


def eval_sighan2015_by_model(correct_fn, sighan_path=sighan_2015_path, verbose=True):
    """
    SIGHAN句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            tgt_pred, pred_detail = correct_fn(src)
            if verbose:
                print()
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)

            # 负样本
            if src == tgt:
                # 预测也为负
                if tgt == tgt_pred:
                    TN += 1
                    print('right')
                # 预测为正
                else:
                    FP += 1
                    print('wrong')
            # 正样本
            else:
                # 预测也为正
                if tgt == tgt_pred:
                    TP += 1
                    print('right')
                # 预测为负
                else:
                    FN += 1
                    print('wrong')
            total_num += 1
        spend_time = time.time() - start_time
        acc = (TP + TN) / total_num
        precision = TP / (TP + FP) if TP > 0 else 0.0
        recall = TP / (TP + FN) if TP > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(
            f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, cost time:{spend_time:.2f} s')
        return acc, precision, recall, f1


if __name__ == "__main__":
    # 评估规则方法的纠错准召率
    # eval_corpus500_by_model(pycorrector.correct)

    # 评估macbert模型的纠错准召率
    from pycorrector.macbert.macbert_corrector import MacBertCorrector

    model = MacBertCorrector()
    eval_corpus500_by_model(model.macbert_correct)
    eval_sighan2015_by_model(model.macbert_correct)

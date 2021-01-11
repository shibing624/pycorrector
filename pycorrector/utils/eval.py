# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:

import copy
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
sighan_2015_path = os.path.join(pwd_path, '../data/cn/sighan_2015/train.tsv')


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


def eval_sighan_corpus(pkl_path, verbose=False):
    sighan_data = load_pkl(pkl_path)
    total_count = 0
    right_count = 0
    for error_sentence, details in sighan_data:
        ids = []
        error_word = ''
        right_word = ''
        if not details:
            continue
        for detail in details:
            idx = detail[0]
            error_word = detail[1]
            right_word = detail[2]
            begin_idx = idx - 1
            ids.append(begin_idx)
        correct_sentence = error_sentence.replace(error_word, right_word)
        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        pred_sentence, pred_detail = pycorrector.correct(error_sentence)
        if pred_sentence == correct_sentence:
            right_count += 1
        else:
            if verbose:
                print('truth:', correct_sentence, details)
                print('predict:', pred_sentence, pred_detail)
        total_count += 1
    right_rate = 0.0
    if total_count > 0:
        right_rate = right_count / total_count
    return right_rate


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


def eval_corpus500_by_rule(input_eval_path=eval_data_path, output_eval_path='', verbose=True):
    res = []
    corpus = load_json(input_eval_path)
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    for data_dict in corpus:
        text = data_dict.get('text', '')
        correction = data_dict.get('correction', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        pred_sentence, pred_detail = pycorrector.correct(text)
        # compute recall
        if errors:
            recall_total_count += 1
            if errors and pred_detail and correction == pred_sentence:
                recall_right_count += 1

        # compute precision
        if correction == pred_sentence:
            right_count += 1
        else:
            err_data_dict = copy.deepcopy(data_dict)
            err_data_dict['pred_sentence'] = pred_sentence
            err_data_dict['pred_errors'] = str(pred_detail)
            res.append(err_data_dict)
            if verbose:
                print("\nwrong:")
                print('input  :', text)
                print('truth  :', correction, errors)
                print('predict:', pred_sentence, pred_detail)
        total_count += 1
    spend_time = time.time() - start_time
    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))
    if output_eval_path:
        save_json(res, output_eval_path)


def eval_corpus500_by_bert(input_eval_path=eval_data_path, output_eval_path='', verbose=True):
    from pycorrector.bert.bert_corrector import BertCorrector
    model = BertCorrector()
    res = []
    corpus = load_json(input_eval_path)
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    for data_dict in corpus:
        text = data_dict.get('text', '')
        correction = data_dict.get('correction', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        pred_sentence, pred_detail = model.bert_correct(text)
        # compute recall
        if errors:
            recall_total_count += 1
            if errors and pred_detail and correction == pred_sentence:
                recall_right_count += 1

        # compute precision
        if correction == pred_sentence:
            right_count += 1
            print("\nright:")
            print('truth  :', text, errors)
            print('predict:', pred_sentence, pred_detail)
        else:
            err_data_dict = copy.deepcopy(data_dict)
            err_data_dict['pred_sentence'] = pred_sentence
            err_data_dict['pred_errors'] = str(pred_detail)
            res.append(err_data_dict)
            if verbose:
                print("\nwrong:")
                print('input  :', text)
                print('truth  :', correction, errors)
                print('predict:', pred_sentence, pred_detail)
        total_count += 1
    spend_time = time.time() - start_time
    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))
    if output_eval_path:
        save_json(res, output_eval_path)


def eval_corpus500_by_ernie(input_eval_path=eval_data_path, output_eval_path='', verbose=True):
    from pycorrector.ernie.ernie_corrector import ErnieCorrector
    model = ErnieCorrector()
    res = []
    corpus = load_json(input_eval_path)
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    for data_dict in corpus:
        text = data_dict.get('text', '')
        correction = data_dict.get('correction', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        pred_sentence, pred_detail = model.ernie_correct(text)
        # compute recall
        if errors:
            recall_total_count += 1
            if errors and pred_detail and correction == pred_sentence:
                recall_right_count += 1

        # compute precision
        if correction == pred_sentence:
            right_count += 1
            print("\nright:")
            print('truth  :', text, errors)
            print('predict:', pred_sentence, pred_detail)
        else:
            err_data_dict = copy.deepcopy(data_dict)
            err_data_dict['pred_sentence'] = pred_sentence
            err_data_dict['pred_errors'] = str(pred_detail)
            res.append(err_data_dict)
            if verbose:
                print("\nwrong:")
                print('input  :', text)
                print('truth  :', correction, errors)
                print('predict:', pred_sentence, pred_detail)
        total_count += 1
    spend_time = time.time() - start_time
    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))
    if output_eval_path:
        save_json(res, output_eval_path)


def eval_sighan_2015_by_rule(sighan_path=sighan_2015_path, verbose=True, num_limit_lines=100):
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            if 0 < num_limit_lines < total_count:
                continue
            src = parts[0]
            trg = parts[1]

            pred, pred_detail = pycorrector.correct(src)

            if src != trg:
                recall_total_count += 1

            if pred == trg:
                right_count += 1
                if src != trg:
                    recall_right_count += 1
                if verbose:
                    print("\nright:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            else:
                if verbose:
                    print("\nwrong:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            total_count += 1

    spend_time = time.time() - start_time

    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))


def eval_sighan_2015_by_bert(sighan_path=sighan_2015_path, verbose=True, num_limit_lines=100):
    from pycorrector.bert.bert_corrector import BertCorrector
    model = BertCorrector()
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            if 0 < num_limit_lines < total_count:
                continue
            src = parts[0]
            trg = parts[1]

            pred, pred_detail = model.bert_correct(src)

            if src != trg:
                recall_total_count += 1

            if pred == trg:
                right_count += 1
                if src != trg:
                    recall_right_count += 1
                if verbose:
                    print("\nright:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            else:
                if verbose:
                    print("\nwrong:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            total_count += 1

    spend_time = time.time() - start_time

    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))


def eval_sighan_2015_by_ernie(sighan_path=sighan_2015_path, verbose=True, num_limit_lines=100):
    from pycorrector.ernie.ernie_corrector import ErnieCorrector
    model = ErnieCorrector()
    total_count = 0
    right_count = 0
    right_rate = 0.0
    recall_rate = 0.0
    recall_right_count = 0
    recall_total_count = 0
    start_time = time.time()
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            if 0 < num_limit_lines < total_count:
                continue
            src = parts[0]
            trg = parts[1]

            pred, pred_detail = model.ernie_correct(src)

            if src != trg:
                recall_total_count += 1

            if pred == trg:
                right_count += 1
                if src != trg:
                    recall_right_count += 1
                if verbose:
                    print("\nright:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            else:
                if verbose:
                    print("\nwrong:")
                    print(f'input  : {src}\ntruth  : {trg}\npredict: {pred} pred_detail: {pred_detail}')
            total_count += 1

    spend_time = time.time() - start_time
    if total_count > 0:
        right_rate = right_count / total_count
    if recall_total_count > 0:
        recall_rate = recall_right_count / recall_total_count
    print('right_rate:{}, right_count:{}, total_count:{};\n'
          'recall_rate:{}, recall_right_count:{}, recall_total_count:{}, spend_time:{} s'.format(right_rate,
                                                                                                 right_count,
                                                                                                 total_count,
                                                                                                 recall_rate,
                                                                                                 recall_right_count,
                                                                                                 recall_total_count,
                                                                                                 spend_time))


if __name__ == "__main__":
    # 生成评估数据集样本，当前已经生成评估集，可以打开注释生成自己的样本分布
    # build_eval_corpus()

    eval_sighan_2015_by_ernie(sighan_path=sighan_2015_path, verbose=True, num_limit_lines=10)
    # 评估规则方法的纠错准召率
    # eval_corpus500_by_rule(eval_data_path)

    # 评估bert模型的纠错准召率
    # eval_corpus500_by_bert(eval_data_path)

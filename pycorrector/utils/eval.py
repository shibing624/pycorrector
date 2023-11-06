# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
import sys
import time
from codecs import open

sys.path.append("../..")

pwd_path = os.path.abspath(os.path.dirname(__file__))
sighan_2015_path = os.path.join(pwd_path, '../data/sighan2015_test.tsv')


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

            r = correct_fn(src)
            tgt_pred, pred_detail = r['target'], r['errors']
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
            f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
            f'cost time:{spend_time:.2f} s, total num: {total_num}')
        return acc, precision, recall, f1


def eval_sighan2015_by_model_batch(correct_fn, sighan_path=sighan_2015_path, verbose=True):
    """
    SIGHAN句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        sighan_path:
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
    srcs = []
    tgts = []
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

            srcs.append(src)
            tgts.append(tgt)

    res = correct_fn(srcs)
    for each_res, src, tgt in zip(res, srcs, tgts):
        if isinstance(each_res, str):
            tgt_pred = each_res
        elif isinstance(each_res, dict):
            tgt_pred = each_res['target']
        else:
            raise ValueError('correct_fn return type error.')
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt)
            print('predict:', each_res)

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
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}')
    return acc, precision, recall, f1


if __name__ == "__main__":
    # 评估macbert模型的纠错准召率
    from pycorrector.macbert.macbert_corrector import MacBertCorrector

    model = MacBertCorrector()
    eval_sighan2015_by_model(model.correct)

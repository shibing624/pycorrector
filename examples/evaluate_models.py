# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys

sys.path.append("../")

import pycorrector
pwd_path = os.path.abspath(os.path.dirname(__file__))


def demo():
    idx_errors = pycorrector.detect('少先队员因该为老人让坐')
    print(idx_errors)


def main(args):
    if args.data == 'sighan_15' and args.model == 'rule':
        # right_rate:0.1798201798201798, right_count:180, total_count:1001;
        # recall_rate:0.15376676986584106, recall_right_count:149, recall_total_count:969, spend_time:121 s
        from pycorrector.utils.eval import eval_sighan_2015_by_rule
        eval_sighan_2015_by_rule()
    if args.data == 'sighan_15' and args.model == 'bert':
        # right_rate:0.37623762376237624, right_count:38, total_count:101;
        # recall_rate:0.3645833333333333, recall_right_count:35, recall_total_count:96, spend_time:503 s
        from pycorrector.utils.eval import eval_sighan_2015_by_bert
        eval_sighan_2015_by_bert()
    if args.data == 'sighan_15' and args.model == 'macbert':
        from pycorrector.utils.eval import eval_sighan_2015_by_macbert
        eval_sighan_2015_by_macbert()
    if args.data == 'sighan_15' and args.model == 'ernie':
        # right_rate:0.297029702970297, right_count:30, total_count:101;
        # recall_rate:0.28125, recall_right_count:27, recall_total_count:96, spend_time:655 s
        from pycorrector.utils.eval import eval_sighan_2015_by_ernie
        eval_sighan_2015_by_ernie()

    if args.data == 'corpus500' and args.model == 'rule':
        # right_rate:0.486, right_count:243, total_count:500;
        # recall_rate:0.18, recall_right_count:54, recall_total_count:300, spend_time:78 s
        from pycorrector.utils.eval import eval_corpus500_by_rule, eval_data_path
        # 评估规则方法的纠错准召率
        out_file = os.path.join(pwd_path, './eval_corpus_error_by_rule.json')
        eval_corpus500_by_rule(eval_data_path, output_eval_path=out_file)
    if args.data == 'corpus500' and args.model == 'bert':
        # right_rate:0.586, right_count:293, total_count:500;
        # recall_rate:0.35, recall_right_count:105, recall_total_count:300, spend_time:1760 s
        from pycorrector.utils.eval import eval_corpus500_by_bert, eval_data_path
        # 评估bert模型的纠错准召率
        out_file = os.path.join(pwd_path, './eval_corpus_error_by_bert.json')
        eval_corpus500_by_bert(eval_data_path, output_eval_path=out_file)
    if args.data == 'corpus500' and args.model == 'macbert':
        from pycorrector.utils.eval import eval_corpus500_by_macbert, eval_data_path
        out_file = os.path.join(pwd_path, './eval_corpus_error_by_macbert.json')
        eval_corpus500_by_macbert(eval_data_path, output_eval_path=out_file)
    if args.data == 'corpus500' and args.model == 'ernie':
        # right_rate:0.598, right_count:299, total_count:500;
        # recall_rate:0.41333333333333333, recall_right_count:124, recall_total_count:300, spend_time:6960 s
        from pycorrector.utils.eval import eval_corpus500_by_ernie, eval_data_path
        # 评估ernie模型的纠错准召率
        out_file = os.path.join(pwd_path, './eval_corpus_error_by_ernie.json')
        eval_corpus500_by_ernie(eval_data_path, output_eval_path=out_file)


if __name__ == '__main__':
    demo()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sighan_15', help='evaluate dataset, sighan_15/corpus500')
    parser.add_argument('--model', type=str, default='rule', help='which model to evaluate, rule/bert/macbert/ernie')
    args = parser.parse_args()
    main(args)

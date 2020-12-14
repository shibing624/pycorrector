# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("../")

import pycorrector

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    idx_errors = pycorrector.detect('少先队员因该为老人让坐')
    print(idx_errors)

    # right_rate:0.1188118811881188, right_count:12, total_count:101;
    # recall_rate:0.07291666666666667, recall_right_count:7, recall_total_count:96, spend_time:33 s
    from pycorrector.utils.eval import eval_rule_with_sighan_2015
    eval_rule_with_sighan_2015()

    # right_rate:0.37623762376237624, right_count:38, total_count:101;
    # recall_rate:0.3645833333333333, recall_right_count:35, recall_total_count:96, spend_time:503 s
    from pycorrector.utils.eval import eval_bert_with_sighan_2015
    eval_bert_with_sighan_2015()

    # right_rate:0.297029702970297, right_count:30, total_count:101;
    # recall_rate:0.28125, recall_right_count:27, recall_total_count:96, spend_time:655 s
    from pycorrector.utils.eval import eval_ernie_with_sighan_2015
    eval_ernie_with_sighan_2015()

    # right_rate:0.486, right_count:243, total_count:500;
    # recall_rate:0.18, recall_right_count:54, recall_total_count:300, spend_time:78 s
    from pycorrector.utils.eval import eval_corpus, eval_data_path
    # 评估规则方法的纠错准召率
    out_file = os.path.join(pwd_path, './eval_corpus_error_by_rule.json')
    eval_corpus(eval_data_path, output_eval_path=out_file)

    # right_rate:0.586, right_count:293, total_count:500;
    # recall_rate:0.35, recall_right_count:105, recall_total_count:300, spend_time:1760 s
    from pycorrector.utils.eval import eval_corpus_by_bert, eval_data_path
    # 评估bert模型的纠错准召率
    out_file = os.path.join(pwd_path, './eval_corpus_error_by_bert.json')
    eval_corpus_by_bert(eval_data_path, output_eval_path=out_file)

    #
    from pycorrector.utils.eval import eval_corpus_by_ernie, eval_data_path
    # 评估ernie模型的纠错准召率
    out_file = os.path.join(pwd_path, './eval_corpus_error_by_ernie.json')
    eval_corpus_by_ernie(eval_data_path, output_eval_path=out_file)

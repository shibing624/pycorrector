# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:

import os
import sys

sys.path.append("../")

pwd_path = os.path.abspath(os.path.dirname(__file__))


# right_rate:0.17065868263473055, right_count:399, total_count:2338;
# recall_rate:0.14633068081343945, recall_right_count:331, recall_total_count:2262, spend_time:4.128874719142914 min
def test_eval_rule_with_sighan_2015():
    from pycorrector.utils.eval import eval_sighan_2015_by_rule
    eval_sighan_2015_by_rule(num_limit_lines=-1)


# right_rate:0.37623762376237624, right_count:38, total_count:101;
# recall_rate:0.3541666666666667, recall_right_count:34, recall_total_count:96, spend_time:550.4112601280212 s
def test_eval_bert_with_sighan_2015():
    from pycorrector.utils.eval import eval_sighan_2015_by_bert
    eval_sighan_2015_by_bert(num_limit_lines=100)


# right_rate:0.297029702970297, right_count:30, total_count:101;
# recall_rate:0.2708333333333333, recall_right_count:26, recall_total_count:96, spend_time:928.6698520183563 s
def test_eval_ernie_with_sighan_2015():
    from pycorrector.utils.eval import eval_sighan_2015_by_ernie
    eval_sighan_2015_by_ernie(num_limit_lines=100)


# right_rate:0.486, right_count:243, total_count:500;
# recall_rate:0.18,recall_right_count:54,recall_total_count:300
def test_eval_rule_with_500():
    from pycorrector.utils.eval import eval_corpus500_by_rule
    # 评估规则方法的纠错准召率
    eval_corpus500_by_rule()


# right_rate:0.58, right_count:290, total_count:500;
# recall_rate:0.37333333333333335,recall_right_count:112,recall_total_count:300
def test_eval_bert_with_500():
    from pycorrector.utils.eval import eval_corpus500_by_bert
    # 评估bert模型的纠错准召率
    eval_corpus500_by_bert()


# right_rate:0.598, right_count:299, total_count:500;
# recall_rate:0.41333333333333333, recall_right_count:124, recall_total_count:300, spend_time:6960 s
def test_eval_ernie_with_500():
    from pycorrector.utils.eval import eval_corpus500_by_ernie
    # 评估bert模型的纠错准召率
    eval_corpus500_by_ernie()

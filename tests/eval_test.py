# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:

import os
import sys

sys.path.append("../")
import pycorrector

pwd_path = os.path.abspath(os.path.dirname(__file__))


# right_rate:0.1798201798201798, right_count:180, total_count:1001;
# recall_rate:0.15376676986584106, recall_right_count:149, recall_total_count:969, spend_time:121.65223574638367 s
def test_eval_rule_with_sighan_2015():
    from pycorrector.utils.eval import eval_sighan2015_by_model

    eval_sighan2015_by_model(pycorrector.correct)


# right_rate:0.486, right_count:243, total_count:500;
# recall_rate:0.18,recall_right_count:54,recall_total_count:300
def test_eval_rule_with_500():
    from pycorrector.utils.eval import eval_corpus500_by_model
    # 评估规则方法的纠错准召率
    eval_corpus500_by_model(pycorrector.correct)

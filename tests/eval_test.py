# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append("../")
from pycorrector import Corrector
from pycorrector.utils.eval import eval_sighan2015_by_model


# right_rate:0.1798201798201798, right_count:180, total_count:1001;
# recall_rate:0.15376676986584106, recall_right_count:149, recall_total_count:969, spend_time:121.65223574638367 s
def test_eval_kenlm_with_sighan_2015():
    m = Corrector()
    eval_sighan2015_by_model(m.correct)

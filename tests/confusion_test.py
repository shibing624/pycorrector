# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")
import os
from pycorrector.utils.io_utils import load_pkl

pwd_path = os.path.abspath(os.path.dirname(__file__))
clp_path = os.path.join(pwd_path, '../pycorrector/data/cn/clp14_C1.pkl')
sighan_path = os.path.join(pwd_path, '../pycorrector/data/cn/sighan15_A2.pkl')


def test_build_confusion_dict():
    confusions = []
    sighan_data = load_pkl(clp_path)
    for error_sentence, right_detail in sighan_data:
        if right_detail:
            if right_detail[0][1:] not in confusions:
                confusions.append(right_detail[0][1:])

    sighan_data = load_pkl(sighan_path)
    for error_sentence, right_detail in sighan_data:
        if right_detail:
            if right_detail[0][1:] not in confusions:
                confusions.append(right_detail[0][1:])
    with open('a.txt', 'w', encoding='utf-8') as f:
        for i in confusions:
            f.write(i[0] + '\t' + i[1] + '\n')

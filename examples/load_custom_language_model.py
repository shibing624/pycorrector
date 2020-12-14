# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("../")
from pycorrector import Corrector

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    lm_path = os.path.join(pwd_path, '../pycorrector/data/people_chars_lm.klm')
    model = Corrector(language_model_path=lm_path)

    corrected_sent, detail = model.correct('少先队员因该为老人让坐')
    print(corrected_sent, detail)

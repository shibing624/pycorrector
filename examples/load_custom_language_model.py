# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
from pycorrector import Corrector
import os
pwd_path = os.path.abspath(os.path.dirname(__file__))
# 使用三元文法语言模型（people_chars.klm）纠错效果更好：
new_language_model_path = os.path.join(pwd_path,'../pycorrector/data/kenlm/people_chars_lm.klm')
model = Corrector()
if os.path.exists(new_language_model_path):
    model.set_language_model_path(new_language_model_path)

error_sentences = [
    '少先队员因该为老人让坐',
    '天地无垠大，我们的舞台无线大',
]
for line in error_sentences:
    correct_sent = model.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

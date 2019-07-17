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
model = Corrector()

error_sentences = [
    '少先队员因该为老人让坐',
    '天地无垠大，我们的舞台无线大',
    '我的形像代言人',
    '我的形像坏人吗',
    '这么做为了谁？',
    '做为一个男人'
]
for line in error_sentences:
    correct_sent = model.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../..")
from pycorrector import Corrector

if __name__ == '__main__':
    model = Corrector(language_model_path='people2014corpus_chars.klm')
    print(model.correct('少先队员因该为老人让坐'))

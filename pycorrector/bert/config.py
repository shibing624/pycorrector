# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

bert_model_dir = os.path.join(pwd_path, '../data/bert_models/chinese_finetuned_lm')
bert_model_vocab = os.path.join(pwd_path, '../data/bert_models/finetuned_chinese_lm/vocab.txt')
output_dir = os.path.join(pwd_path, './output')
max_seq_length = 128
threshold = 0.1

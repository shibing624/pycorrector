# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# bert_model_dir = os.path.join(pwd_path, '../data/bert_pytorch/chinese_finetuned_lm')
bert_model_dir = os.path.join(pwd_path, '../data/bert_pytorch/multi_cased_L-12_H-768_A-12')
# bert_model_vocab = os.path.join(pwd_path, '../data/bert_pytorch/chinese_finetuned_lm/vocab.txt')
bert_model_vocab = os.path.join(pwd_path, '../data/bert_pytorch/multi_cased_L-12_H-768_A-12/vocab.txt')
output_dir = os.path.join(pwd_path, './output')
predict_file = os.path.join(pwd_path, '../data/bert_pytorch/samples/sample_text.txt')
max_seq_length = 384
do_lower_case = True

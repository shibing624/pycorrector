# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

bert_model_dir = os.path.join(pwd_path, '../data/bert_models/chinese_finetuned_lm/')
bert_model_path = os.path.join(pwd_path, '../data/bert_models/chinese_finetuned_lm/pytorch_model.bin')
bert_config_path = os.path.join(pwd_path, '../data/bert_models/chinese_finetuned_lm/config.json')

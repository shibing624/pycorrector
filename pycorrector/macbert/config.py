# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: Use CGED corpus
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# CGED chinese corpus
cged_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

sighan_train_path = os.path.join(pwd_path, '../data/cn/sighan_2015/train.tsv')

use_segment = True
segment_type = 'char'

output_dir = os.path.join(pwd_path, 'output')

dataset = 'sighan'  # 'sighan' or 'cged'

# config
model_dir = os.path.join(output_dir, 'model_{}'.format(dataset))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

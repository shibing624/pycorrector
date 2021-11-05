# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

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

use_segment = False
segment_type = 'char'

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'dev.json')
# Validation data path.
valid_path = ''
test_path = os.path.join(output_dir, 'dev.json')

dataset = 'sighan'  # 'sighan' or 'cged'

pretrained_model = "hfl/chinese-macbert-base"  # official macbert pretrained model
# config
src_vocab_path = os.path.join(output_dir, 'vocab_source.txt')
trg_vocab_path = os.path.join(output_dir, 'vocab_target.txt')
ckpt_path = os.path.join(output_dir, 'macbert4csc_model_{}.ckpt'.format(dataset))

batch_size = 32
test_batch_size = 8
epochs = 10
max_length = 128
gpu_ids = [0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
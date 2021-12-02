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

use_segment = True
segment_type = 'char'

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

dataset = 'sighan'  # 'sighan' or 'cged'
arch = 'bertseq2seq'  # 'seq2seq' or 'convseq2seq' or 'bertseq2seq'

model_name_or_path = "bert-base-chinese"  # for bertseq2seq

# config
src_vocab_path = os.path.join(output_dir, 'vocab_source.txt')
trg_vocab_path = os.path.join(output_dir, 'vocab_target.txt')
model_dir = os.path.join(output_dir, 'model_{}'.format(dataset))

batch_size = 32
epochs = 40  # bertseq2seq is '40', other is '200'
max_length = 128

gpu_id = 0
dropout = 0.25
embed_size = 128
hidden_size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

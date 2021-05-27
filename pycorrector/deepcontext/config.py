# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: network configuration
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# chinese corpus
cged_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

sighan_train_path = os.path.join(pwd_path, '../data/cn/sighan_2015/train.tsv')

use_segment = True
segment_type = 'char'
dataset = 'sighan'  # 'sighan' or 'cged'

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
model_dir = os.path.join(output_dir, 'models')
model_path = os.path.join(model_dir, 'model.pth')
vocab_path = os.path.join(model_dir, 'vocab.txt')

# nets
word_embed_size = 200
hidden_size = 200
n_layers = 2
dropout = 0.0

# train
epochs = 20
batch_size = 64
min_freq = 1
learning_rate = 1e-3

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

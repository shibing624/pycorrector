# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

# CGED chinese corpus
raw_train_paths = [
    # '../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
    # '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    '../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
output_dir = 'output'
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

# seq2seq_attn_train config
vocab_json_path = os.path.join(output_dir, 'seq2seq_config.json')
attn_model_path = os.path.join(output_dir, 'attn_model.h5')

# config
batch_size = 64
epochs = 100
rnn_hidden_dim = 128
maxlen = 400

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

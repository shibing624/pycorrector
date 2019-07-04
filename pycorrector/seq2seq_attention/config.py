# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

# seq2seq_attn_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
attn_model_path = os.path.join(output_dir, 'attn_model.weight')

# config
batch_size = 64
epochs = 40
rnn_hidden_dim = 128
maxlen = 400
dropout = 0.0
use_gpu = False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

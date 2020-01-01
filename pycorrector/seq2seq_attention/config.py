# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# toy chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

# seq2seq_attn_train config
save_src_vocab_path = os.path.join(output_dir, 'vocab_source.txt')
save_trg_vocab_path = os.path.join(output_dir, 'vocab_target.txt')
model_dir = os.path.join(output_dir, 'models')

batch_size = 32
epochs = 10
embedding_dim = 256
hidden_dim = 256
maxlen = 32
gpu_id = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
model_dir = os.path.join(pwd_path, 'output/model')
# Training data path.
src_train_path = os.path.join(output_dir, 'src-train.txt')
tgt_train_path = os.path.join(output_dir, 'tgt-train.txt')
# Validation data path.
src_test_path = os.path.join(output_dir, 'src-test.txt')
tgt_test_path = os.path.join(output_dir, 'tgt-test.txt')

vocab_path = os.path.join(output_dir, 'vocab.txt')

maximum_length = 50
shuffle_buffer_size = 10000
gradients_accum = 8
train_steps = 10000
save_every = 1000
report_every = 50

batch_size = 32
beam_size = 4

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

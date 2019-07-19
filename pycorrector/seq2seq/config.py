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
val_path = os.path.join(output_dir, 'test.txt')
test_path = os.path.join(output_dir, 'test.txt')

vocab_path = os.path.join(output_dir, 'vocab.txt')
vocab_max_size = 50000
vocab_min_count = 5

batch_size = 32
epochs = 50
gpu_id = 0
save_model_batch_num = 100

# Path of the model saved
save_model_dir = os.path.join(output_dir, 'models')

model_path = os.path.join(save_model_dir, 'seq2seq_49_0.model')

predict_out_path = os.path.join(output_dir, 'test_out.txt')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

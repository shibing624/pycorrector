# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

# CGED chinese corpus
raw_train_paths = [
    # '../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
    # '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    # '../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
output_dir = './output'
train_path = output_dir + '/train.txt'  # Training data path.
test_path = output_dir + '/test.txt'  # Validation data path.

# config
batch_size = 128
epochs = 10
rnn_hidden_dim = 200
save_model_path = output_dir + '/cged_seq2seq_model.h5'  # Path of the model saved

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

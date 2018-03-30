# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus

# CGED chinese corpus
train_path = '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'  # Training data path.
val_path = '../data/cn/CGED/CGED18_HSK_TestingSet.xml'  # Validation data path.
test_path = '../data/cn/CGED/CGED18_HSK_TestingSet.xml'

model_path = './output/cged_model'  # Path of the model saved, default is output_path/model
enable_special_error = False
num_steps = 3000  # Number of steps to train.
decode_sentence = False  # Whether we should decode sentences of the user.

# Config
buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]  # use a number of buckets and pad to the closest one for efficiency.
steps_per_checkpoint = 100
max_steps = 10000
max_vocab_size = 10000
size = 512
num_layers = 4
max_gradient_norm = 5.0
batch_size = 128
learning_rate = 0.5
learning_rate_decay_factor = 0.99
use_lstm = False
use_rms_prop = False

enable_decode_sentence = False  # Test with input error sentence
enable_test_decode = True  # Test with test set

import os

if not os.path.exists(model_path):
    os.makedirs(model_path)

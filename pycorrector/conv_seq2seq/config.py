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
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
trainpref = os.path.join(output_dir, 'train')
train_src_path = os.path.join(output_dir, 'train.src')
train_trg_path = os.path.join(output_dir, 'train.trg')

# Validation data path.
valpref = os.path.join(output_dir, 'valid')
val_src_path = os.path.join(output_dir, 'valid.src')
val_trg_path = os.path.join(output_dir, 'valid.trg')

# Path of the fairseq data saved
data_bin_dir = os.path.join(output_dir, 'bin')

test_path = os.path.join(output_dir, 'valid.src')

vocab_path = os.path.join(output_dir, 'vocab.txt')
vocab_max_size = 6000

arch = 'fconv'
batch_size = 64
max_len = 400
# epochs = 50


# (float, optional): temperature, where values >1.0 produce more uniform samples
# and values <1.0 produce sharper samples (default: 1.0)
temperature = 0.8

# Path of the model saved
save_model_dir = os.path.join(output_dir, 'models')
best_model_path = os.path.join(save_model_dir, 'checkpoint_best.pt')

predict_out_path = os.path.join(output_dir, 'valid.src.predict')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

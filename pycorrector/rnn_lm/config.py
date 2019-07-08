# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os


pwd_path = os.path.abspath(os.path.dirname(__file__))

# CGED chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]
output_dir = os.path.join(pwd_path, 'output')
train_word_path = output_dir + '/train_words.txt'
# vocab
word_dict_path = output_dir + '/word_freq.txt'
model_dir = output_dir + '/model'
# config
cutoff_frequency = 5
batch_size = 64
learning_rate = 0.01

model_prefix = 'lm'
num_save_epochs = 6
epochs = 20

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

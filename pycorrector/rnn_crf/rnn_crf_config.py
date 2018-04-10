# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os

# CGED chinese corpus
train_paths = ['../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
               # '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
               # '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'
               ]
word_data_path = 'output/words.txt'
label_data_path = 'output/labels.txt'
word_dict_path = 'output/word_dict.txt'
label_dict_path = 'output/label_dict.txt'

# Config
batch_size = 64
epoch = 1
embedding_dim = 100
rnn_hidden_dim = 200
cutoff_frequency = 0
pred_save_path = 'output/pred.txt'
model_path = './output'  # Path of the model saved, default is output_path/model

if not os.path.exists(model_path):
    os.makedirs(model_path)

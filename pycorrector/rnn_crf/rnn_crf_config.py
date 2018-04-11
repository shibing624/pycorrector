# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
output_dir = './output'
# CGED chinese corpus
train_paths = ['../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
               # '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
               # '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'
               ]
train_word_path = output_dir + '/train_words.txt'
train_label_path = output_dir + '/train_labels.txt'
test_paths = ['../data/cn/CGED/CGED18_HSK_TestingSet.xml',
               # '../data/cn/CGED/CGED17_HSK_TestingSet.xml',
               # '../data/cn/CGED/CGED16_HSK_TestingSet.xml'
               ]
test_word_path = output_dir + '/test_words.txt'
test_label_path = output_dir + '/test_labels.txt'
# vocab
word_dict_path = output_dir + '/word_dict.txt'
label_dict_path = output_dir + '/label_dict.txt'

# config
batch_size = 64
epoch = 1
embedding_dim = 100
rnn_hidden_dim = 200
cutoff_frequency = 0
save_model_path = output_dir + '/rnn_crf_model.h5'  # Path of the model saved, default is output_path/model

# infer
save_pred_path = output_dir + '/pred.txt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

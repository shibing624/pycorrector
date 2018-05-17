# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os

output_dir = './output'

# CGED chinese corpus
train_paths = ['../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
               '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
               '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
               # '../data/cn/CGED/sample_HSK_TrainingSet.xml',
               ]
train_word_path = output_dir + '/train_words.txt'
# vocab
word_dict_path = output_dir + '/word_dict.txt'

# config
start_token = 'B'
end_token = 'E'
batch_size = 64
learning_rate = 0.01
model_dir = output_dir + '/model'
model_prefix = 'lm'
num_save_epochs = 6
epochs = 50

# infer
gen_file_path = output_dir + '/gen.txt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

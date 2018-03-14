# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: configuration
import os

word_freq_path = 'data/word_dict.txt'
word_freq_model_path = 'data/word_dict.pkl'

char_file_path = 'data/char_set.txt'

same_pinyin_text_path = 'data/same_pinyin.txt'
same_pinyin_model_path = 'data/same_pinyin.pkl'

same_stroke_text_path = 'data/same_stroke.txt'
same_stroke_model_path = 'data/same_stroke.pkl'

# path of training data
train_data_path = "data/rank/train.txt"
# path of testing data, if testing file does not exist,
# testing will not be performed at the end of each training pass
test_data_path = "data/rank/test.txt"
# path of word dictionary, if this file does not exist,
# word dictionary will be built from training data.
dic_path = "data/rank/vocab.txt"

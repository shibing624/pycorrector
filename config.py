# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: configuration
import os

################## for language model  ##################

bimodel_path = './data/model/zhwiki_bigram.klm'
trimodel_path = './data/model/zhwiki_trigram.klm'

################## for training  #########################
# path of training data
train_data_path = "data/rank/train.txt"
# path of testing data, if testing file does not exist,
# testing will not be performed at the end of each training pass
test_data_path = "data/rank/test.txt"
# path of word dictionary, if this file does not exist,
# word dictionary will be built from training data.
dic_path = "data/rank/vocab.txt"


# directory to save the trained model
# create a new directory if the directoy does not exist
# model_save_dir = "model"
# if not os.path.exists(model_save_dir):
#     os.mkdir(model_save_dir)

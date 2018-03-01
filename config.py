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

share_semantic_generator = True  # whether to share network parameters between source and target
share_embed = True  # whether to share word embedding between source and target

num_workers = 1 # threads
use_gpu = False  # to use gpu or not

num_batches_to_log = 50
num_batches_to_save_model = 400  # number of batches to output model

# directory to save the trained model
# create a new directory if the directoy does not exist
model_save_dir = "output"



if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
import os

# CGED chinese corpus
raw_train_paths = [
    # '../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
    # '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
    # '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    '../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
output_dir = './output'
train_path = output_dir + '/train.txt'  # Training data path.
test_path = output_dir + '/test.txt'  # Validation data path.

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

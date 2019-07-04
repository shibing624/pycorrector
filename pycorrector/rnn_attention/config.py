# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]
output_dir = os.path.join(pwd_path, 'output')
train_path = output_dir + '/train.txt'  # Training data path.
test_path = output_dir + '/test.txt'  # Validation data path.

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

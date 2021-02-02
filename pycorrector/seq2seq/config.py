# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: Use CGED corpus
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# CGED chinese corpus
cged_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

sighan_train_path = os.path.join(pwd_path, '../data/cn/sighan_2015/train.tsv')

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

dataset = 'sighan'  # 'sighan' or 'cged'
arch = 'convseq2seq'  # 'seq2seq' or 'convseq2seq'

# config
src_vocab_path = os.path.join(output_dir, 'vocab_source.txt')
trg_vocab_path = os.path.join(output_dir, 'vocab_target.txt')
model_path = os.path.join(output_dir, 'model_{}_{}.pth'.format(dataset, arch))

batch_size = 32
epochs = 200
max_length = 128
gpu_id = 0
dropout = 0.25
embed_size = 128
hidden_size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

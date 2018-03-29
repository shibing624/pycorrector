# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
train_path = '../data/en/fce/fce_train.txt'  # Training data path.
val_path = '../data/en/fce/fce_val.txt'  # Validation data path.
test_path = '../data/en/fce/fce_test.txt'

output_path = './output'  # Path of the model saved, default is output_path/model
enable_data_dropout = False
num_steps = 3000  # Number of steps to train.
decode_sentence = False  # Whether we should decode sentences of the user.

# FCEConfig
buckets = [(10, 10), (15, 15), (20, 20), (40, 40)]  # use a number of buckets and pad to the closest one for efficiency.

steps_per_checkpoint = 20
max_steps = 100

max_vocabulary_size = 10000

size = 128
num_layers = 2
max_gradient_norm = 5.0
batch_size = 64
learning_rate = 0.5
learning_rate_decay_factor = 0.99

use_lstm = False
use_rms_prop = False

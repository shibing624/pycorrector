# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

train_path = sys.argv[1]
words_path = sys.argv[2]
labels_path = sys.argv[3]


def read_2_list(data_path):
    words, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        word_line, label_line = '', ''
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) == 2:
                word_line += parts[0].strip() + ' '
                label_line += parts[1].strip() + ' '
            else:
                words.append(word_line)
                labels.append(label_line)
                word_line, label_line = '', ''
    return words, labels


def write_2_data(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for i in data:
            f.write(i.strip() + '\n')
        print('done to ' + save_path)


words, labels = read_2_list(train_path)
write_2_data(words, words_path)
write_2_data(labels, labels_path)

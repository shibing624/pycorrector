# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 


import sys

words_path = sys.argv[1]
labels_path = sys.argv[2]
words, labels = [], []


def append_2_list(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            data.append(line.split())
    return data


words = append_2_list(words_path)
labels = append_2_list(labels_path)

out_path = sys.argv[3]
with open(out_path, 'w', encoding='utf-8') as f:
    for i in range(len(words)):
        for j in range(len(words[i])):
            f.write(words[i][j] + '\t' + labels[i][j] + '\n')
        f.write('\n')
    print('done to ' + out_path)
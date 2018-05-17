# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import collections

import numpy as np


def process_data(file_name,start_token='B',end_token='E'):
    data = []
    with open(file_name, "r", encoding='utf-8') as f:
        count = 0
        for line in f.readlines():
            try:
                content = line.strip()
                content = content.replace(' ', '')
                if len(content) < 5 or len(content) > 400:
                    continue
                content = start_token + content + end_token
                data.append(content)
                count = count + 1
            except ValueError as e:
                pass

    print('file:', file_name, "size:", count)
    data = sorted(data, key=lambda l: len(l))

    total_words = []
    for line in data:
        total_words += [word for word in line]
    counter = collections.Counter(total_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    words = words[:len(words)] + (' ',)
    word_idx = dict(zip(words, range(len(words))))
    data_vector = [list(map(lambda word: word_idx.get(word, len(words)), i)) for i in data]

    return data_vector, word_idx, words


def generate_batch(batch_size, data_vec, word_to_int):
    n_chunk = len(data_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = data_vec[start_index:end_index]
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches

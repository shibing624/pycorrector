# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import collections

import numpy as np

MIN_LEN = 5
MAX_LEN = 400
START_TOKEN = 'B'
END_TOKEN = 'E'
UNK_TOKEN = 'UNK'


def process_data(file_name, word_dict_path=None, cutoff_frequency=10):
    data = []
    with open(file_name, "r", encoding='utf-8') as f:
        count = 0
        for line in f:
            content = line.strip()
            content = content.replace(' ', '')
            if len(content) < MIN_LEN or len(content) > MAX_LEN:
                continue
            content = START_TOKEN + content + END_TOKEN
            data.append(content)
            count = count + 1

    print('file:', file_name, "size:", count)
    data = sorted(data, key=lambda l: len(l))

    total_words = []
    for line in data:
        total_words += [word for word in line]
    counter = dict()
    for k, v in collections.Counter(total_words).items():
        if v < cutoff_frequency:
            continue
        counter[k] = v
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    vocab, _ = zip(*count_pairs)

    vocab = (UNK_TOKEN,) + vocab[:len(vocab)]
    word_to_int = dict(zip(vocab, range(len(vocab))))
    if word_dict_path:
        save_dict(word_to_int, word_dict_path)
    data_vector = [list(map(lambda word: word_to_int.get(word, 0), i)) for i in data]
    return data_vector, word_to_int


def save_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                print('error', line)
    return dict_data


def generate_batch(batch_size, data_vec, word_to_int):
    n_chunk = len(data_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = data_vec[start_index:end_index]
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_to_int[UNK_TOKEN], np.int32)
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

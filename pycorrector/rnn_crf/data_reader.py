# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence

UNK_ID = 0


def vectorize_data(path, word_dict):
    data_ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            ids = sent2ids(line, word_dict)
            data_ids.append(ids)
    return data_ids


def build_dict(data_path,
               save_path,
               cutoff_frequency=0,
               insert_extra_words=[]):
    values = defaultdict(int)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split()
            for w in line_split:
                values[w] += 1

    with open(save_path, "w", encoding='utf-8') as f:
        for w in insert_extra_words:
            f.write("%s\t%s\n" % (w, UNK_ID))

        for v, count in sorted(
                values.items(), key=lambda x: x[1], reverse=True):
            if count > cutoff_frequency:
                f.write("%s\t%d\n" % (v, count))
        print('save %s data to %s' % (data_path, save_path))


def sent2ids(sent, vocab):
    """
    transform a sentence to a list of ids.
    """
    return [vocab.get(w, UNK_ID) for w in sent.split()]


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, 'r', encoding='utf-8').readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, 'r', encoding='utf-8').readlines()))


def pad_sequence(word_ids, maxlen=300):
    return sequence.pad_sequences(np.array(word_ids), maxlen=maxlen)


def get_max_len(word_ids):
    return max(len(line) for line in word_ids)


def load_test_id(dict_path):
    return [''.join(line.strip().split()) for idx, line in
            enumerate(open(dict_path, 'r', encoding='utf-8').readlines())]

# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Corpus for model

import sys
from codecs import open
from collections import Counter

# Define constants associated with the usual special tokens.
PAD_TOKEN = 'PAD'
GO_TOKEN = 'GO'
EOS_TOKEN = 'EOS'
UNK_TOKEN = 'UNK'


def save_word_dict(dict_data, save_path):
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


def read_vocab(input_texts, max_size=50000, min_count=5):
    token_counts = Counter()
    special_tokens = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, UNK_TOKEN]
    for line in input_texts:
        for char in line.strip():
            char = char.strip()
            if not char:
                continue
            token_counts.update(char)
    # Sort word count by value
    count_pairs = token_counts.most_common()
    vocab = [k for k, v in count_pairs if v >= min_count]
    # Insert the special tokens to the beginning
    vocab[0:0] = special_tokens
    full_token_id = list(zip(vocab, range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    return vocab2id


def read_samples_by_string(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.lower().strip().split('\t')
            if len(parts) != 2:
                print('error ', line)
                continue
            source, target = parts[0], parts[1]
            yield source, target


def build_dataset(path):
    print('Read data, path:{0}'.format(path))
    sources, targets = [], []
    for source, target in read_samples_by_string(path):
        sources.append(source)
        targets.append(target)
    return sources, targets


def show_progress(curr, total, time=""):
    prog_ = int(round(100.0 * float(curr) / float(total)))
    dstr = '[' + '>' * int(round(prog_ / 4)) + ' ' * (25 - int(round(prog_ / 4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time + '\r')
    sys.stdout.flush()


def str2id(s, vocab2id, maxlen):
    # 文字转id
    return [vocab2id.get(c.strip(), vocab2id[UNK_TOKEN]) for c in s[:maxlen] if c.strip()]


def padding(x, vocab2id):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [vocab2id[PAD_TOKEN]] * (ml - len(i)) for i in x]


def id2str(ids, id2vocab):
    # id转文字，找不到的用空字符代替
    return ''.join([id2vocab.get(i, UNK_TOKEN) for i in ids])

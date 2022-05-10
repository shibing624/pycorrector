# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
from codecs import open
from collections import Counter


# Define constants associated with the usual special tokens.
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'


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


def read_vocab(input_texts, max_size=None, min_count=0):
    token_counts = Counter()
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
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


def max_length(tensor):
    return max(len(t) for t in tensor)


def create_dataset(path, num_examples):
    """
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    :param path:
    :param num_examples:
    :return:
    """
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def preprocess_sentence(w):
    w = w.lower().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = SOS_TOKEN + ' ' + w + ' ' + EOS_TOKEN
    return w


def show_progress(curr, total, time=""):
    prog_ = int(round(100.0 * float(curr) / float(total)))
    dstr = '[' + '>' * int(round(prog_ / 4)) + ' ' * (25 - int(round(prog_ / 4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time + '\r')
    sys.stdout.flush()

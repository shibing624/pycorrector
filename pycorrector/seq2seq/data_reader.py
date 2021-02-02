# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: Corpus for model

import sys
from codecs import open
from collections import Counter

import numpy as np

# Define constants associated with the usual special tokens.
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


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
    special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for texts in input_texts:
        for token in texts:
            token_counts.update(token)
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


def create_dataset(path, num_examples=None):
    """
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    :param path:
    :param num_examples:
    :return:
    """
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(s) for s in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def preprocess_sentence(sentence):
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    return [SOS_TOKEN] + sentence.lower().split() + [EOS_TOKEN]


def show_progress(curr, total, time=""):
    prog_ = int(round(100.0 * float(curr) / float(total)))
    dstr = '[' + '>' * int(round(prog_ / 4)) + ' ' * (25 - int(round(prog_ / 4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time + '\r')
    sys.stdout.flush()


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)  # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs, max_length=None):
    if max_length:
        seqs = [seq[:max_length] for seq in seqs]
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(src_sentences, trg_sentences, batch_size, max_length=None):
    minibatches = get_minibatches(len(src_sentences), batch_size)
    examples = []
    for minibatch in minibatches:
        mb_src_sentences = [src_sentences[t] for t in minibatch]
        mb_trg_sentences = [trg_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_src_sentences, max_length)
        mb_y, mb_y_len = prepare_data(mb_trg_sentences, max_length)
        examples.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return examples


def one_hot(src_sentences, trg_sentences, src_dict, trg_dict, sort_by_len=True):
    """vector the sequences.
    """
    out_src_sentences = [[src_dict.get(w, 0) for w in sent] for sent in src_sentences]
    out_trg_sentences = [[trg_dict.get(w, 0) for w in sent] for sent in trg_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # sort length
    if sort_by_len:
        sorted_index = len_argsort(out_src_sentences)
        out_src_sentences = [out_src_sentences[i] for i in sorted_index]
        out_trg_sentences = [out_trg_sentences[i] for i in sorted_index]

    return out_src_sentences, out_trg_sentences

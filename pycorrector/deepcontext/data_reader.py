# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Corpus for model
"""

import json
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
    word_freq = {k: v for k, v in count_pairs if v >= min_count}
    # Insert the special tokens to the beginning
    vocab[0:0] = special_tokens
    full_token_id = list(zip(vocab, range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    special_tokens_dict = {k: 0 for k in special_tokens}
    word_freq.update(special_tokens_dict)
    return vocab2id, word_freq


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)  # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(src_sentences, batch_size):
    minibatches = get_minibatches(len(src_sentences), batch_size)
    examples = []
    for minibatch in minibatches:
        mb_src_sentences = [src_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_src_sentences)
        examples.append((mb_x, mb_x_len))
    return examples


def one_hot(src_sentences, src_dict, sort_by_len=False):
    """vector the sequences.
    """
    out_src_sentences = [[src_dict.get(w, 0) for w in sent] for sent in src_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # sort length
    if sort_by_len:
        sorted_index = len_argsort(out_src_sentences)
        out_src_sentences = [out_src_sentences[i] for i in sorted_index]

    return out_src_sentences


def write_embedding(id2word, nn_embedding, use_cuda, filename):
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(nn_embedding.num_embeddings, nn_embedding.embedding_dim))
        if use_cuda:
            embeddings = nn_embedding.weight.data.cpu().numpy()
        else:
            embeddings = nn_embedding.weight.data.numpy()

        for word_id, vec in enumerate(embeddings):
            word = id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))


def write_config(filename, **kwargs):
    with open(filename, mode='w', encoding='utf-8') as f:
        json.dump(kwargs, f)


def read_config(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return json.load(f)


class Dataset:
    def __init__(
            self,
            train_path,
            batch_size,
            min_freq,
            device,
            vocab_path,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            sos_token=SOS_TOKEN,
            eos_token=EOS_TOKEN,
    ):
        sentences = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().lower().split()
                if len(tokens) > 0:
                    sentences.append([sos_token] + tokens + [eos_token])
        self.sent_dict = self._gathered_by_lengths(sentences)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device

        vocab_2_ids, self.word_freqs = read_vocab(sentences, min_count=min_freq)
        save_word_dict(vocab_2_ids, vocab_path)
        self.vocab_2_ids = load_word_dict(vocab_path)

        self.id_2_vocabs = {v: k for k, v in vocab_2_ids.items()}
        train_vec = one_hot(sentences, self.vocab_2_ids)
        self.train_data = gen_examples(train_vec, batch_size)

        if self.pad_token:
            self.pad_index = self.vocab_2_ids[self.pad_token]

    def _gathered_by_lengths(self, sentences):
        lengths = [(index, len(sent)) for index, sent in enumerate(sentences)]
        lengths = sorted(lengths, key=lambda x: x[1], reverse=True)

        sent_dict = dict()
        current_length = -1
        for (index, length) in lengths:
            if current_length == length:
                sent_dict[length].append(index)
            else:
                sent_dict[length] = [index]
                current_length = length

        return sent_dict

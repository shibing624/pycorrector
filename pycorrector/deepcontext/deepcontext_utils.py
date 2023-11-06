# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import json
import math
from codecs import open
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# Define constants associated with the usual special tokens.
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


class NegativeSampling(nn.Module):
    def __init__(self,
                 embed_size,
                 counter,
                 n_negatives,
                 power,
                 device,
                 ignore_index):
        super(NegativeSampling, self).__init__()

        self.counter = counter
        self.n_negatives = n_negatives
        self.power = power
        self.device = device

        self.W = nn.Embedding(
            num_embeddings=len(counter),
            embedding_dim=embed_size,
            padding_idx=ignore_index
        )
        self.W.weight.data.zero_()
        self.logsigmoid = nn.LogSigmoid()
        self.sampler = WalkerAlias(np.power(counter, power))

    def negative_sampling(self, shape):
        if self.n_negatives > 0:
            return torch.tensor(self.sampler.sample(shape=shape), dtype=torch.long, device=self.device)
        else:
            raise NotImplementedError

    def forward(self, sentence, context):
        batch_size, seq_len = sentence.size()
        emb = self.W(sentence)
        pos_loss = self.logsigmoid((emb * context).sum(2))

        neg_samples = self.negative_sampling(shape=(batch_size, seq_len, self.n_negatives))
        neg_emb = self.W(neg_samples)
        neg_loss = self.logsigmoid((-neg_emb * context.unsqueeze(2)).sum(3)).sum(2)
        return -(pos_loss + neg_loss).sum()


class WalkerAlias:
    """
    This is from Chainer's implementation.
    You can find the original code at
    https://github.com/chainer/chainer/blob/v4.4.0/chainer/utils/walker_alias.py
    This class is
        Copyright (c) 2015 Preferred Infrastructure, Inc.
        Copyright (c) 2015 Preferred Networks, Inc.
    """

    def __init__(self, probs):
        prob = np.array(probs, np.float32)
        prob /= np.sum(prob)
        threshold = np.ndarray(len(probs), np.float32)
        values = np.ndarray(len(probs) * 2, np.int32)
        il, ir = 0, 0
        pairs = list(zip(prob, range(len(probs))))
        pairs.sort()
        for prob, i in pairs:
            p = prob * len(probs)
            while p > 1 and ir < il:
                values[ir * 2 + 1] = i
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = i
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert ((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values

    def sample(self, shape):
        ps = np.random.uniform(0, 1, shape)
        pb = ps * len(self.threshold)
        index = pb.astype(np.int32)
        left_right = (self.threshold[index] < pb - index).astype(np.int32)
        return self.values[index * 2 + left_right]


class Context2vec(nn.Module):
    def __init__(
            self,
            vocab_size,
            counter,
            word_embed_size,
            hidden_size,
            n_layers,
            use_mlp,
            dropout,
            pad_index,
            device,
            is_inference
    ):
        super(Context2vec, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_mlp = use_mlp
        self.device = device
        self.is_inference = is_inference
        self.rnn_output_size = hidden_size

        self.drop = nn.Dropout(dropout)
        self.l2r_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_size,
            padding_idx=pad_index
        )
        self.l2r_rnn = nn.LSTM(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.r2l_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_size,
            padding_idx=pad_index
        )
        self.r2l_rnn = nn.LSTM(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.criterion = NegativeSampling(
            hidden_size,
            counter,
            ignore_index=pad_index,
            n_negatives=10,
            power=0.75,
            device=device
        )

        if use_mlp:
            self.MLP = MLP(
                input_size=hidden_size * 2,
                mid_size=hidden_size * 2,
                output_size=hidden_size,
                dropout=dropout
            )
        else:
            self.weights = nn.Parameter(torch.zeros(2, hidden_size))
            self.gamma = nn.Parameter(torch.ones(1))

        self.init_weights()

    def init_weights(self):
        std = math.sqrt(1. / self.word_embed_size)
        self.r2l_emb.weight.data.normal_(0, std)
        self.l2r_emb.weight.data.normal_(0, std)

    def forward(self, sentences, target, target_pos=None):
        # input: <BOS> a b c <EOS>
        # reversed_sentences: <EOS> c b a
        # sentences: <BOS> a b c

        reversed_sentences = sentences.flip(1)[:, :-1]
        sentences = sentences[:, :-1]

        l2r_emb = self.l2r_emb(sentences)
        r2l_emb = self.r2l_emb(reversed_sentences)

        output_l2r, _ = self.l2r_rnn(l2r_emb)
        output_r2l, _ = self.r2l_rnn(r2l_emb)

        # output_l2r: h(<BOS>)   h(a)     h(b)
        # output_r2l:     h(b)   h(c) h(<EOS>)

        output_l2r = output_l2r[:, :-1, :]
        output_r2l = output_r2l[:, :-1, :].flip(1)

        if self.is_inference:
            # user_input: I like [] .
            # target_pos: 2 (starts from zero)

            # output_l2r:   h(<BOS>)      h(I)     h(like)      h([])
            # output_r2l:    h(like)     h([])        h(.)   h(<EOS>)
            # output_l2r[target_pos]: h(like)
            # output_r2l[target_pos]:    h(.)

            if self.use_mlp:
                output_l2r = output_l2r[0, target_pos]
                output_r2l = output_r2l[0, target_pos]
                c_i = self.MLP(torch.cat((output_l2r, output_r2l), dim=0))
            return c_i
        else:
            # on a training phase
            if self.use_mlp:
                c_i = self.MLP(torch.cat((output_l2r, output_r2l), dim=2))
            else:
                s_task = torch.nn.functional.softmax(self.weights, dim=0)
                c_i = torch.stack((output_l2r, output_r2l), dim=2) * s_task
                c_i = self.gamma * c_i.sum(2)

            loss = self.criterion(target, c_i)
            return loss

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))

    def run_inference(self, input_tokens, target, target_pos, k=10):
        context_vector = self.forward(input_tokens, target=None, target_pos=target_pos)
        if target is None:
            topv, topi = ((self.criterion.W.weight * context_vector).sum(dim=1)).data.topk(k)
            return topv, topi
        else:
            context_vector /= torch.norm(context_vector, p=2)
            target_vector = self.criterion.W.weight[target]
            target_vector /= torch.norm(target_vector, p=2)
            similarity = (target_vector * context_vector).sum()
            return similarity.item()

    def norm_embedding_weight(self, embedding_module):
        embedding_module.weight.data /= torch.norm(embedding_module.weight.data, p=2, dim=1, keepdim=True)
        # replace NaN with zero
        embedding_module.weight.data[embedding_module.weight.data != embedding_module.weight.data] = 0


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            mid_size,
            output_size,
            n_layers=2,
            dropout=0.3,
            activation_function='relu'
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.mid_size = mid_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(input_size, output_size))
        else:
            self.MLP.append(nn.Linear(input_size, mid_size))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(mid_size, mid_size))
            self.MLP.append(nn.Linear(mid_size, output_size))

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x
        for i in range(self.n_layers - 1):
            out = self.MLP[i](self.drop(out))
            out = self.activation_function(out)
        return self.MLP[-1](self.drop(out))


def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            items = line.split('\t')
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                logger.warning(f"IndexError: {line}")
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
    vocab = special_tokens + vocab
    if max_size is not None:
        vocab = vocab[:max_size]
    vocab2id = dict(zip(vocab, range(len(vocab))))
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


def prepare_data(seqs, max_length=512):
    seqs = [seq[:max_length] for seq in seqs]
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)

    x = np.zeros((n_samples, max_length)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(src_sentences, batch_size, max_length):
    minibatches = get_minibatches(len(src_sentences), batch_size)
    examples = []
    for minibatch in minibatches:
        mb_src_sentences = [src_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_src_sentences, max_length)
        examples.append((mb_x, mb_x_len))
    return examples


def one_hot(src_sentences, src_dict, sort_by_len=False):
    """vector the sequences."""
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


class ContextDataset:
    def __init__(
            self,
            train_path,
            batch_size=64,
            max_length=512,
            min_freq=0,
            device='cuda',
            vocab_path='vocab.txt',
            vocab_max_size=50000,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            sos_token=SOS_TOKEN,
            eos_token=EOS_TOKEN,
    ):
        sentences = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = list(line.strip().lower())
                if len(tokens) > 0:
                    sentences.append([sos_token] + tokens + [eos_token])
        self.sent_dict = self._gathered_by_lengths(sentences)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device

        self.vocab_2_ids, self.word_freqs = read_vocab(sentences, max_size=vocab_max_size, min_count=min_freq)
        logger.debug(f"vocab_2_ids size: {len(self.vocab_2_ids)}, word_freqs: {len(self.word_freqs)}, "
                     f"vocab_2_ids head: {list(self.vocab_2_ids.items())[:10]}, "
                     f"word_freqs head: {list(self.word_freqs.items())[:10]}")
        save_word_dict(self.vocab_2_ids, vocab_path)

        self.id_2_vocabs = {v: k for k, v in self.vocab_2_ids.items()}
        self.train_data = gen_examples(one_hot(sentences, self.vocab_2_ids), batch_size, max_length)
        self.pad_index = self.vocab_2_ids.get(self.pad_token, 0)

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

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: loss function
"""
import math

import numpy as np
import torch
import torch.nn as nn


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

        self.W = nn.Embedding(num_embeddings=len(counter),
                              embedding_dim=embed_size,
                              padding_idx=ignore_index)
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


class WalkerAlias(object):
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

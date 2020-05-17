# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: build torchtext dataset
"""

import numpy as np
from torchtext import data


class Dataset(object):
    def __init__(self,
                 sentences: list,
                 batch_size: int,
                 min_freq: int,
                 device: int,
                 pad_token='<PAD>',
                 unk_token='<UNK>',
                 bos_token='<BOS>',
                 eos_token='<EOS>',
                 seed=777):

        np.random.seed(seed)
        self.sent_dict = self._gathered_by_lengths(sentences)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.device = device

        self.sentence_field = data.Field(use_vocab=True,
                                         unk_token=self.unk_token,
                                         pad_token=self.pad_token,
                                         init_token=self.bos_token,
                                         eos_token=self.eos_token,
                                         batch_first=True,
                                         include_lengths=False)
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)

        self.sentence_field.build_vocab(sentences, min_freq=min_freq)
        self.vocab = self.sentence_field.vocab
        if self.pad_token:
            self.pad_index = self.sentence_field.vocab.stoi[self.pad_token]

        self.dataset = self._create_dataset(self.sent_dict, sentences)

    def get_raw_sentence(self, sentences):
        return [[self.vocab.itos[idx] for idx in sentence]
                for sentence in sentences]

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

    def _create_dataset(self, sent_dict, sentences):
        datasets = dict()
        _fields = [('sentence', self.sentence_field),
                   ('id', self.sentence_id_field)]
        for sent_length, sent_indices in sent_dict.items():
            sent_indices = np.array(sent_indices)
            items = [*zip(sentences[sent_indices], sent_indices[:, np.newaxis])]
            datasets[sent_length] = data.Dataset(self._get_examples(items, _fields), _fields)
        return np.random.permutation(list(datasets.values()))

    def _get_examples(self, items: list, fields: list):
        return [data.Example.fromlist(item, fields) for item in items]

    def get_batch_iter(self, batch_size: int):

        def sort(data: data.Dataset) -> int:
            return len(getattr(data, 'sentence'))

        for dataset in self.dataset:
            yield data.Iterator(dataset=dataset,
                                batch_size=batch_size,
                                sort_key=sort,
                                train=True,
                                repeat=False,
                                device=self.device
                                )

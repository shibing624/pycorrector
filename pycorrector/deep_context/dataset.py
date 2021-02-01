# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: build torchtext dataset
"""

from pycorrector.deep_context.data_reader import PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, read_vocab, save_word_dict, \
    load_word_dict, one_hot, gen_examples


class Dataset(object):
    def __init__(self,
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

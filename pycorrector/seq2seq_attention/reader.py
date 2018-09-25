# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Date reader for training seq2seq model
from collections import Counter

# Define constants associated with the usual special tokens.
PAD_ID = 0
GO_ID = 1
EOS_ID = 2

PAD_TOKEN = 'PAD'
EOS_TOKEN = 'EOS'
GO_TOKEN = 'GO'


class Reader(object):
    def __init__(self, train_path=None, token_2_id=None,
                 special_tokens=()):
        if train_path is None:
            self.token_2_id = token_2_id
        else:
            token_counts = Counter()
            for tokens in self.read_tokens(train_path):
                token_counts.update(tokens)

            self.token_counts = token_counts
            # Get max_vocabulary size words
            count_pairs = sorted(token_counts.items(), key=lambda k: (-k[1], k[0]))
            vocab, _ = list(zip(*count_pairs))
            vocab = list(vocab)
            # Insert the special tokens to the beginning
            vocab[0:0] = special_tokens
            full_token_id = list(zip(vocab, range(len(vocab))))
            self.full_token_2_id = dict(full_token_id)
            self.token_2_id = dict(full_token_id)
        self.id_2_token = {v: k for k, v in self.token_2_id.items()}

    def read_tokens(self, path):
        """
        Reads the given file line by line and yields the list of tokens present
        in each line.

        :param path:
        :return:
        """
        raise NotImplementedError("Must implement read_tokens")

    def unknown_token(self):
        raise NotImplementedError("Must implement unknow_tokens")

    def read_samples_by_string(self, path):
        """
        Reads the given file line by line and yields the word-form of each
        derived sample.

        :param path:
        :return:
        """
        raise NotImplementedError("Must implement read_samples")

    def convert_token_2_id(self, token):
        """
        Token to id
        :param token:
        :return:
        """
        token_id = token if token in self.token_2_id else self.unknown_token()
        return self.token_2_id[token_id]

    def convert_id_2_token(self, id):
        """
        Word id to token
        :param id:
        :return:
        """
        return self.id_2_token[id]

    def is_unknown_token(self, token):
        """
        True if the given token is out of vocabulary
        :param token:
        :return:
        """
        return token not in self.token_2_id or token == self.unknown_token()

    def sentence_2_token_ids(self, sentence):
        """
        Convert a sentence to word ids
        :param sentence:
        :return:
        """
        return [self.convert_token_2_id(w) for w in sentence.split()]

    def token_ids_2_tokens(self, word_ids):
        """
        Convert a list of word ids to words
        :param word_ids:
        :return:
        """
        return [self.convert_id_2_token(w) for w in word_ids]

    def read_samples(self, path):
        """
        Read sample of path's data
        :param path:
        :return: generate list
        """
        for source_words, target_words in self.read_samples_by_string(path):
            source = [self.convert_token_2_id(w) for w in source_words]
            target = [self.convert_token_2_id(w) for w in target_words]
            # head: "GO"; last: "EOS"
            target.insert(0, GO_ID)
            target.append(EOS_ID)
            yield source, target

    def read_samples_tokens(self, path):
        """
        Read sample of path's data
        :param path:
        :return: generate list
        """
        for source_words, target_words in self.read_samples_by_string(path):
            target = target_words
            # head: "GO"; last: "EOS"
            target.insert(0, GO_TOKEN)
            target.append(EOS_TOKEN)
            yield source_words, target

    def build_dataset(self, path):
        print('Read data, path:{0}'.format(path))
        sources, targets = [], []
        for source, target in self.read_samples_tokens(path):
            sources.append(source)
            targets.append(target)
        return sources, targets

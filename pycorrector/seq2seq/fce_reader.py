# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import random
from reader import Reader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN


class FCEReader(Reader):
    """
    Read FCE data set
    """
    UNKNOWN_TOKEN = 'UNK'
    DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}
    REPLACEMENTS = {"there": "their", "their": "there", "then": "than", "than": "then"}

    def __init__(self, config, train_path=None, token_2_id=None,
                 dropout_prob=0.25, replacement_prob=0.25, dataset_copies=2):
        super(FCEReader, self).__init__(
            config, train_path=train_path, token_2_id=token_2_id,
            special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN, FCEReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)
        self.dropout_prob = dropout_prob
        self.replacement_prob = replacement_prob
        self.UNKNOW_ID = self.token_2_id[FCEReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        for tokens in self.read_tokens(path):
            source = []
            target = []
            for token in tokens:
                target.append(token)
                if self.config.enable_data_dropout:
                    # Random dropout words from the input
                    dropout_token = (token in FCEReader.DROPOUT_TOKENS and random.random() < self.dropout_prob)
                    replace_token = (token in FCEReader.REPLACEMENTS and random.random() < self.replacement_prob)
                    if replace_token:
                        source.append(FCEReader.REPLACEMENTS[tokens])
                    elif not dropout_token:
                        source.append(token)
                else:
                    source.append(token)
            yield source, target

    def unknown_token(self):
        return FCEReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        i = 0
        with open(path, 'r', 'utf-8') as f:
            for line in f:
                if line in f:
                    if i % 2 == 1:
                        yield line.lower()[5:].strip().split()
                    i += 1

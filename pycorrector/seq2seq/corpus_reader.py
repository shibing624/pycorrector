# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Corpus for model
import random

from reader import Reader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN
from xml.dom import minidom
from utils.text_utils import segment


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
        self.UNKNOWN_ID = self.token_2_id[FCEReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                line_src = f.readline()
                line_dst = f.readline()
                if not line_src or len(line_src) < 5:
                    break
                source = line_src.lower()[5:].strip().split()
                target = line_dst.lower()[5:].strip().split()
                if self.config.enable_special_error:
                    new_source = []
                    for token in source:
                        # Random dropout words from the input
                        dropout_token = (token in FCEReader.DROPOUT_TOKENS and
                                         random.random() < self.dropout_prob)
                        replace_token = (token in FCEReader.REPLACEMENTS and
                                         random.random() < self.replacement_prob)
                        if replace_token:
                            new_source.append(FCEReader.REPLACEMENTS[source])
                        elif not dropout_token:
                            new_source.append(token)
                    source = new_source
                yield source, target

    def unknown_token(self):
        return FCEReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Input the correct text, which start with 0
                if i % 2 == 1:
                    if line and len(line) > 5:
                        yield line.lower()[5:].strip().split()
                i += 1


class CGEDReader(Reader):
    """
    Read CGED data set
    """
    UNKNOWN_TOKEN = 'UNK'

    def __init__(self, config, train_path=None, token_2_id=None, dataset_copies=2):
        super(CGEDReader, self).__init__(
            config, train_path=train_path, token_2_id=token_2_id,
            special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN, CGEDReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)
        self.UNKNOWN_ID = self.token_2_id[CGEDReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            dom_tree = minidom.parse(f)
            docs = dom_tree.documentElement.getElementsByTagName('DOC')
            for doc in docs:
                source_text = doc.getElementsByTagName('TEXT')[0]. \
                    childNodes[0].data.strip()
                target_text = doc.getElementsByTagName('CORRECTION')[0]. \
                    childNodes[0].data.strip()
                source = segment(source_text, cut_type='char')
                target = segment(target_text, cut_type='char')
                yield source, target

    def unknown_token(self):
        return CGEDReader.UNKNOWN_TOKEN

    def read_tokens(self, path, is_infer=False):
        with open(path, 'r', encoding='utf-8') as f:
            dom_tree = minidom.parse(f)
            docs = dom_tree.documentElement.getElementsByTagName('DOC')
            for doc in docs:
                if is_infer:
                    # Input the error text
                    sentence = doc.getElementsByTagName('TEXT')[0]. \
                        childNodes[0].data.strip()
                else:
                    # Input the correct text
                    sentence = doc.getElementsByTagName('CORRECTION')[0]. \
                        childNodes[0].data.strip()
                yield segment(sentence, cut_type='char')

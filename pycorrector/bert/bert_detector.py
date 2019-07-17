# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: use bert detect chinese char error
"""
import sys
import time

import numpy as np
import torch
from pytorch_transformers import BertForMaskedLM
from pytorch_transformers import BertTokenizer

sys.path.append('../..')
from pycorrector.detector import ErrorType
from pycorrector.utils.logger import logger
from pycorrector.bert import config


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids,
                 segment_ids=None,
                 mask_ids=None,
                 masked_lm_labels=None,
                 input_tokens=None,
                 id=None,
                 token=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.masked_lm_labels = masked_lm_labels
        self.input_tokens = input_tokens
        self.id = id
        self.token = token


class BertDetector(object):
    def __init__(self, bert_model_dir=config.bert_model_dir,
                 bert_model_vocab=config.bert_model_vocab,
                 threshold=0.1):
        self.name = 'bert_detector'
        self.bert_model_dir = bert_model_dir
        self.bert_model_vocab = bert_model_vocab
        self.initialized_bert_detector = False
        self.threshold = threshold

    def check_bert_detector_initialized(self):
        if not self.initialized_bert_detector:
            self.initialize_bert_detector()

    def initialize_bert_detector(self):
        t1 = time.time()
        self.bert_tokenizer = BertTokenizer(vocab_file=self.bert_model_vocab)
        self.MASK_TOKEN = "[MASK]"
        self.MASK_ID = self.bert_tokenizer.convert_tokens_to_ids([self.MASK_TOKEN])[0]
        # Prepare model
        self.model = BertForMaskedLM.from_pretrained(self.bert_model_dir)
        logger.debug("Loaded model ok, path: %s, spend: %.3f s." % (self.bert_model_dir, time.time() - t1))
        self.initialized_bert_detector = True

    def _convert_sentence_to_detect_features(self, sentence):
        """Loads a sentence into a list of `InputBatch`s."""
        self.check_bert_detector_initialized()
        features = []
        tokens = self.bert_tokenizer.tokenize(sentence)
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        for idx, token_id in enumerate(token_ids):
            masked_lm_labels = [-1] * len(token_ids)
            masked_lm_labels[idx] = token_id
            features.append(
                InputFeatures(input_ids=token_ids,
                              masked_lm_labels=masked_lm_labels,
                              input_tokens=tokens,
                              id=idx,
                              token=tokens[idx]))
        return features

    def predict_token_prob(self, sentence):
        self.check_bert_detector_initialized()
        result = []
        eval_features = self._convert_sentence_to_detect_features(sentence)

        for f in eval_features:
            input_ids = torch.tensor([f.input_ids])
            masked_lm_labels = torch.tensor([f.masked_lm_labels])
            outputs = self.model(input_ids, masked_lm_labels=masked_lm_labels)
            masked_lm_loss, predictions = outputs[:2]
            prob = np.exp(-masked_lm_loss.item())
            result.append([prob, f])
        return result

    def detect(self, sentence):
        """
        句子改错
        :param sentence: 句子文本
        :param threshold: 阈值
        :return: list[list], [error_word, begin_pos, end_pos, error_type]
        """
        maybe_errors = []
        for prob, f in self.predict_token_prob(sentence):
            logger.debug('prob:%s, token:%s, idx:%s' % (prob, f.token, f.id))
            if prob < self.threshold:
                maybe_errors.append([f.token, f.id, f.id + 1, ErrorType.char])
        return maybe_errors


if __name__ == "__main__":
    d = BertDetector()

    error_sentences = ['少先队员因该为老人让座',
                       '少先队员因该为老人让坐',
                       '少 先 队 员 因 该 为老人让座',
                       '少 先 队 员 因 该 为老人让坐',
                       '机七学习是人工智能领遇最能体现智能的一个分支',
                       '机七学习是人工智能领遇最能体现智能的一个分知']
    t1 = time.time()
    for sent in error_sentences:
        err = d.detect(sent)
        print("original sentence:{} => detect sentence:{}".format(sent, err))

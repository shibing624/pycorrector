# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: use bert correct chinese char error
"""
import operator
import sys
import time

import torch
from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

sys.path.append('../..')
from pycorrector.bert import config
from pycorrector.bert.bert_detector import BertDetector, InputFeatures
from pycorrector.detector import ErrorType
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.logger import logger

MASK_TOKEN = "[MASK]"


class BertCorrector(BertDetector):
    def __init__(self, bert_model_dir='',
                 bert_model_vocab='',
                 max_seq_length=128,
                 predict_batch_size=8,
                 max_predictions_per_seq=20,
                 threshold=0.001):
        super(BertCorrector, self).__init__(bert_model_dir=bert_model_dir,
                                            bert_model_vocab=bert_model_vocab,
                                            max_seq_length=max_seq_length,
                                            predict_batch_size=predict_batch_size,
                                            max_predictions_per_seq=max_predictions_per_seq,
                                            threshold=threshold)
        self.name = 'bert_corrector'
        self.bert_model_dir = bert_model_dir
        self.bert_model_vocab = bert_model_vocab
        self.max_seq_length = max_seq_length
        self.initialized_bert_corrector = False

    def check_bert_corrector_initialized(self):
        if not self.initialized_bert_corrector:
            self.initialize_bert_corrector()

    def initialize_bert_corrector(self):
        t1 = time.time()
        self.bert_tokenizer = BertTokenizer(self.bert_model_vocab)
        self.MASK_ID = self.bert_tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
        # Prepare model
        self.model = BertForMaskedLM.from_pretrained(self.bert_model_dir)
        logger.debug("Loaded model ok, path: %s, spend: %.3f s." % (self.bert_model_dir, time.time() - t1))
        self.initialized_bert_corrector = True

    def convert_sentence_to_features(self, sentence, max_seq_length, error_begin_idx, error_end_idx):
        """Loads a sentence into a list of `InputBatch`s."""
        self.check_bert_corrector_initialized()
        features = []
        tokens_a = list(sentence)

        # For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0      0   0   0  0    0   0
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        k = error_begin_idx + 1
        for i in range(error_end_idx - error_begin_idx):
            tokens[k] = '[MASK]'
            k += 1
        segment_ids = [0] * len(tokens)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        mask_ids = [i for i, v in enumerate(input_ids) if v == self.MASK_ID]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # Original:
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          mask_ids=mask_ids,
                          segment_ids=segment_ids,
                          input_tokens=tokens))

        return features

    def check_vocab_has_all_token(self, sentence):
        self.check_bert_corrector_initialized()
        flag = True
        for i in list(sentence):
            if i not in self.bert_tokenizer.vocab:
                flag = False
                break
        return flag

    def predict_mask_token(self, sentence, error_begin_idx, error_end_idx):
        self.check_bert_corrector_initialized()
        corrected_item = sentence[error_begin_idx:error_end_idx]
        eval_features = self.convert_sentence_to_features(
            sentence=sentence,
            max_seq_length=self.max_seq_length,
            error_begin_idx=error_begin_idx,
            error_end_idx=error_end_idx
        )

        for f in eval_features:
            input_ids = torch.tensor([f.input_ids])
            segment_ids = torch.tensor([f.segment_ids])
            predictions = self.model(input_ids, segment_ids)
            # confirm we were able to predict 'henson'
            masked_ids = f.mask_ids
            if masked_ids:
                for idx, i in enumerate(masked_ids):
                    predicted_index = torch.argmax(predictions[0, i]).item()
                    predicted_token = self.bert_tokenizer.convert_ids_to_tokens([predicted_index])[0]
                    logger.debug('original text is: %s' % f.input_tokens)
                    logger.debug('Mask predict is: %s' % predicted_token)
                    corrected_item = predicted_token
        return corrected_item

    def correct(self, sentence=''):
        """
        句子改错
        :param sentence: 句子文本
        :return: 改正后的句子, list(wrong, right, begin_idx, end_idx)
        """
        detail = []
        maybe_errors = self.detect(sentence)
        for item, begin_idx, end_idx, err_type in maybe_errors:
            # 纠错，逐个处理
            before_sent = sentence[:begin_idx]
            after_sent = sentence[end_idx:]

            if err_type == ErrorType.char:
                # 对非中文的错字不做处理
                if not is_chinese_string(item):
                    continue
                if not self.check_vocab_has_all_token(sentence):
                    continue
                # 取得所有可能正确的字
                corrected_item = self.predict_mask_token(sentence, begin_idx, end_idx)
            elif err_type == ErrorType.word:
                corrected_item = item
            else:
                print('not strand error_type')
            # output
            if corrected_item != item:
                sentence = before_sent + corrected_item + after_sent
                detail_word = [item, corrected_item, begin_idx, end_idx]
                detail.append(detail_word)
        detail = sorted(detail, key=operator.itemgetter(2))
        return sentence, detail


if __name__ == "__main__":
    bertCorrector = BertCorrector(config.bert_model_dir,
                                  config.bert_model_vocab,
                                  config.max_seq_length)

    error_sentences = ['少先队员因该为老人让座',
                       '少先队员因该为老人让坐',
                       '机七学习是人工智能领遇最能体现智能的一个分支',
                       '机七学习是人工智能领遇最能体现智能的一个分知']
    for error_sentence in error_sentences:
        correct_sent = bertCorrector.correct(error_sentence)
        print("original sentence:{} => correct sentence:{}".format(error_sentence, correct_sent))

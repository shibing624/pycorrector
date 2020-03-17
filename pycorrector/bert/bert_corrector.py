# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: use bert detect and correct chinese char error
"""

import sys
import time

from transformers import pipeline

sys.path.append('../..')
from pycorrector.bert import config
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector


class BertCorrector(Corrector):
    def __init__(self, bert_model_dir=config.bert_model_dir,
                 bert_config_path=config.bert_config_path,
                 bert_model_path=config.bert_model_path):
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
        self.mask = '[MASK]'
        t1 = time.time()
        self.model = pipeline('fill-mask',
                              model=bert_model_path,
                              config=bert_config_path,
                              tokenizer=bert_model_dir)
        logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))

    def bert_correct(self, sentence):
        """
        句子纠错
        :param sentence: 句子文本
        :return: list[list], [error_word, begin_pos, end_pos, error_type]
        """
        maybe_errors = []
        for idx, s in enumerate(sentence):
            # 对非中文的错误不做处理
            if not is_chinese_string(s):
                continue

            sentence_lst = list(sentence)
            sentence_lst[idx] = self.mask
            sentence_new = ''.join(sentence_lst)
            predicts = self.model(sentence_new)
            top_tokens = []
            for p in predicts:
                token_id = p.get('token', 0)
                token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                top_tokens.append(token_str)

            if top_tokens and (s not in top_tokens):
                # 取得所有可能正确的词
                candidates = self.generate_items(s)
                if not candidates:
                    continue
                for token_str in top_tokens:
                    if token_str in candidates:
                        maybe_errors.append([s, token_str, idx, idx + 1])
                        break
        return maybe_errors


if __name__ == "__main__":
    d = BertCorrector()
    error_sentences = [
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        err = d.bert_correct(sent)
        print("original sentence:{} => detect sentence:{}".format(sent, err))

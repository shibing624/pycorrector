# -*- coding: utf-8 -*-
"""
@Time   :   2021-02-03 18:05:54
@File   :   macbert_corrector.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import operator
import sys
import time

# import kenlm  # import kenlm before torch, when torch>=1.7.1

sys.path.append('../..')
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector
from pycorrector.macbert.correction_pipeline import CorrectionPipeline
from pycorrector import config
from pycorrector.transformers import BertTokenizer, BertForMaskedLM


class MacBertCorrector(Corrector):
    def __init__(self, bert_model_dir=config.macbert_model_dir):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        t1 = time.time()
        bert_model = BertForMaskedLM.from_pretrained(bert_model_dir)
        tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
        self.model = CorrectionPipeline(
            task='correction',
            model=bert_model,
            tokenizer=tokenizer,
            device=0,  # gpu device id
        )
        if self.model:
            self.mask = self.model.tokenizer.mask_token
            logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))

    def macbert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        details = []
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = self.split_text_by_maxlen(text, maxlen=128)
        blocks = [block[0] for block in blocks]
        results = self.model(blocks)
        text_new = ''.join([rst['corrected_text'] for rst in results])
        for i, ori_char in enumerate(text):
            if ori_char == ' ':
                # pipeline 处理后的 text_new 不含空格，在此处补充空格。
                text_new = text_new[:i] + ' ' + text_new[i:]
                continue
            if i >= len(text_new):
                continue
            if ori_char != text_new[i]:
                details.append([ori_char, text_new[i], i, i + 1])

        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    d = MacBertCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.macbert_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))

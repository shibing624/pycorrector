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
import os
from transformers import BertTokenizer, BertForMaskedLM

sys.path.append('../..')
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.macbert.correction_pipeline import CorrectionPipeline
from pycorrector import config

from pycorrector.utils.tokenizer import split_text_by_maxlen


class MacBertCorrector(object):
    def __init__(self, macbert_model_dir=config.macbert_model_dir):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        t1 = time.time()
        if not os.path.exists(os.path.join(macbert_model_dir, 'vocab.txt')):
            macbert_model_dir = "shibing624/macbert4csc-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(macbert_model_dir)
        # tokenizer.add_tokens(['“', '”'])
        macbert_model = BertForMaskedLM.from_pretrained(macbert_model_dir)
        # macbert_model.resize_token_embeddings(len(tokenizer))
        self.model = CorrectionPipeline(
            task='correction',
            model=macbert_model,
            tokenizer=tokenizer,
            device=0,  # gpu device id
        )
        if self.model:
            logger.debug('Loaded macbert model: %s, spend: %.3f s.' % (macbert_model_dir, time.time() - t1))

    def macbert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        details = []
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
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
                details.append((ori_char, text_new[i], i, i + 1))

        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    m = MacBertCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
        '他们的吵翻很不错，再说他们做的咖喱鸡也好吃',
        '影像小孩子想的快，学习管理的斑法',
        '餐厅的换经费产适合约会',
        '走路真的麻坊，我也没有喝的东西，在家汪了',
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '妈妈说："别趴地上了，快起来，你还吃饭吗？"，我说："好。"就扒起来了。',
        '你说：“怎么办？”我怎么知道？',
        '我父母们常常说：“那时候吃的东西太少，每天只能吃一顿饭。”想一想，人们都快要饿死，谁提出化肥和农药的污染。',
        '这本新书《居里夫人传》将的很生动有趣',
        '我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……',
    ]
    for sent in error_sentences:
        corrected_sent, err = m.macbert_correct(sent)
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import operator
import sys
import time
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch

sys.path.append('../..')
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector import config

from pycorrector.utils.tokenizer import split_text_by_maxlen

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MacBertCorrector(object):
    def __init__(self, macbert_model_dir=config.macbert_model_dir, device=-1):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        t1 = time.time()
        if not os.path.exists(os.path.join(macbert_model_dir, 'vocab.txt')):
            macbert_model_dir = "shibing624/macbert4csc-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(macbert_model_dir)
        self.model = BertForMaskedLM.from_pretrained(macbert_model_dir)
        self.unk_tokens = [' ', '“', '”', '‘', '’', '琊']
        logger.debug('Loaded macbert model: %s, spend: %.3f s.' % (macbert_model_dir, time.time() - t1))

    def macbert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        blocks = [block[0] for block in blocks]
        outputs = self.model(**self.tokenizer(blocks, padding=True, return_tensors='pt'))

        def get_errors(corrected_text, origin_text):
            sub_details = []
            for i, ori_char in enumerate(origin_text):
                if ori_char in self.unk_tokens:
                    # add blank space
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                    continue
                if i >= len(corrected_text):
                    continue
                if ori_char != corrected_text[i]:
                    sub_details.append((ori_char, corrected_text[i], i, i + 1))
            sub_details = sorted(sub_details, key=operator.itemgetter(2))
            return corrected_text, sub_details

        for ids, text in zip(outputs.logits, blocks):
            _text = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = _text[:len(text)]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            details.extend(sub_details)
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

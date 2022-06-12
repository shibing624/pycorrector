# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import operator
import sys
import time
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

sys.path.append('../..')
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector import config
from pycorrector.utils.tokenizer import split_text_by_maxlen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class T5Corrector(object):
    def __init__(self, model_dir=config.t5_model_dir):
        super(T5Corrector, self).__init__()
        self.name = 'byt5_corrector'
        t1 = time.time()
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/byt5-small-chinese-correction"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded byt5 correction model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def t5_correct(self, text, max_length=128):
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
        blocks = split_text_by_maxlen(text, maxlen=max_length)
        block_texts = [block[0] for block in blocks]
        inputs = self.tokenizer(block_texts, padding=True, max_length=max_length, truncation=True,
                                return_tensors='pt').to(device)
        outputs = self.model.generate(**inputs, max_length=max_length)

        for text, idx in blocks:
            decode_tokens = self.tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '').replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            details.extend(sub_details)
        return text_new, details


if __name__ == "__main__":
    m = T5Corrector('./output/byt5-small-chinese-correction/')
    error_sentences = [
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
    ]
    for sent in error_sentences:
        corrected_sent, err = m.t5_correct(sent)
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))

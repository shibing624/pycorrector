# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use Electra model detect and correct chinese char error
"""

import operator
import sys
import time

import torch

sys.path.append('../..')
from pycorrector.transformers import pipeline, ElectraForPreTraining

from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector
from pycorrector.utils.tokenizer import split_2_short_text
from pycorrector import config


class ElectraCorrector(Corrector):
    def __init__(self, d_model_dir=config.electra_D_model_dir, g_model_dir=config.electra_G_model_dir):
        super(ElectraCorrector, self).__init__()
        self.name = 'electra_corrector'
        t1 = time.time()
        self.g_model = pipeline(
            "fill-mask",
            model=g_model_dir,
            tokenizer=g_model_dir,
            device=0,  # gpu device id
        )
        self.d_model = ElectraForPreTraining.from_pretrained(d_model_dir)

        if self.g_model:
            self.mask = self.g_model.tokenizer.mask_token
            logger.debug('Loaded electra model: %s, spend: %.3f s.' % (g_model_dir, time.time() - t1))

    def electra_detect(self, sentence):
        fake_inputs = self.g_model.tokenizer.encode(sentence, return_tensors="pt")
        discriminator_outputs = self.d_model(fake_inputs)
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

        error_ids = []
        for index, s in enumerate(predictions.tolist()[0][1:-1]):
            if s > 0.0:
                error_ids.append(index)
        return error_ids

    def electra_correct(self, text):
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
        blocks = split_2_short_text(text, include_symbol=True)
        for blk, start_idx in blocks:
            error_ids = self.electra_detect(blk)
            sentence_lst = list(blk)
            for idx in error_ids:
                s = sentence_lst[idx]
                if is_chinese_string(s):
                    # 处理中文错误
                    sentence_lst[idx] = self.mask
                    sentence_new = ''.join(sentence_lst)
                    # 生成器fill-mask预测[mask]，默认取top5
                    predicts = self.g_model(sentence_new)
                    top_tokens = []
                    for p in predicts:
                        token_id = p.get('token', 0)
                        token_str = self.g_model.tokenizer.convert_ids_to_tokens(token_id)
                        top_tokens.append(token_str)

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    sentence_lst[idx] = token_str
                                    break
                    # 还原
                    if sentence_lst[idx] == self.mask:
                        sentence_lst[idx] = s

            blk_new = ''.join(sentence_lst)
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    m = ElectraCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '今天新情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = m.electra_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))

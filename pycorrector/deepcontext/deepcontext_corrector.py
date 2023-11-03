# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import operator
import os
import sys
import time

import numpy as np
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L
from loguru import logger
sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.corrector import Corrector
from pycorrector.utils.tokenizer import segment, split_text_by_maxlen

pwd_path = os.path.abspath(os.path.dirname(__file__))

D.guard().__enter__()  # activate paddle `dygrpah` mode


class ErnieCloze(ErnieModelForPretraining):
    def __init__(self, *args, **kwargs):
        super(ErnieCloze, self).__init__(*args, **kwargs)
        del self.pooler_heads

    def forward(self, src_ids, *args, **kwargs):
        pooled, encoded = ErnieModel.forward(self, src_ids, *args, **kwargs)
        # paddle ernie mask_id = 3, mask_token = [MASK]
        mask_id = 3
        encoded_2d = L.gather_nd(encoded, L.where(src_ids == mask_id))
        encoded_2d = self.mlm(encoded_2d)
        encoded_2d = self.mlm_ln(encoded_2d)
        logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
        return logits_2d


class ErnieCorrector(Corrector):
    def __init__(self, model_dir='ernie-1.0', topN=5):
        super(ErnieCorrector, self).__init__()
        self.name = 'ernie_corrector'
        t1 = time.time()
        self.ernie_tokenizer = ErnieTokenizer.from_pretrained(model_dir)
        self.rev_dict = {v: k for k, v in self.ernie_tokenizer.vocab.items()}
        self.rev_dict[self.ernie_tokenizer.pad_id] = ''  # replace [PAD]
        self.rev_dict[self.ernie_tokenizer.sep_id] = ''  # replace [PAD]
        self.rev_dict[self.ernie_tokenizer.unk_id] = ''  # replace [PAD]
        self.cloze = ErnieCloze.from_pretrained(model_dir)
        self.cloze.eval()
        logger.debug('Loaded ernie model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))
        self.mask_id = self.ernie_tokenizer.mask_id  # 3
        self.mask_token = self.rev_dict[self.mask_id]  # "[MASK]"
        logger.debug('ernie mask_id :{}, mask_token: {}'.format(self.mask_id, self.mask_token))
        self.topN = topN

    def predict_mask(self, sentence_with_mask):
        ids, id_types = self.ernie_tokenizer.encode(sentence_with_mask)
        ids = np.expand_dims(ids, 0)
        ids = D.to_variable(ids)
        logits = self.cloze(ids).numpy()
        output_ids = np.argsort(logits, -1)
        masks_ret = []
        # 倒序，取最可能预测词topN
        for masks in output_ids:
            temp = []
            for mask in masks[::-1][:self.topN]:
                temp.append(self.rev_dict[mask])
            masks_ret.append(temp)
        # 处理多个mask的情况
        out = []
        for i in range(len(masks_ret)):
            temp = []
            for j in range(len(masks_ret[i])):
                temp.append(masks_ret[i][j])
            out.append(temp)
        # transpose out data
        # [['智', '学', '能', '技', '互'], ['能', '慧', '习', '智', '商']] => 智能 学慧 能习 技智 互商
        out = np.transpose(np.array(out))
        out = [''.join(i) for i in out.tolist()]
        ret = []
        for i in out:
            ret.append({'token': i})
        return ret

    def ernie_correct(self, text, ernie_cut_type='char'):
        """
        句子纠错
        :param text: 句子文本
        :param ernie_cut_type: 切词类型（char/word）
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=512)
        for blk, start_idx in blocks:
            blk_new = ''
            blk = segment(blk, cut_type=ernie_cut_type, pos=False)
            for idx, s in enumerate(blk):
                # 处理中文错误
                if is_chinese_string(s):
                    sentence_lst = blk[:idx] + blk[idx:]
                    sentence_lst[idx] = self.mask_token * len(s)
                    sentence_new = ' '.join(sentence_lst)
                    # 预测，默认取top5
                    predicts = self.predict_mask(sentence_new)
                    top_tokens = [p.get('token', '') for p in predicts]
                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append((s, token_str, start_idx + idx, start_idx + idx + 1))
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


import operator
import sys
import time

import torch
from transformers import pipeline, ElectraForPreTraining
from loguru import logger
sys.path.append('../..')

from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.corrector import Corrector
from pycorrector.utils.tokenizer import split_2_short_text
from pycorrector import config

device_id = 0 if torch.cuda.is_available() else -1


class ElectraCorrector(Corrector):
    def __init__(self, d_model_dir=config.electra_D_model_dir, g_model_dir=config.electra_G_model_dir,
                 device=device_id):
        super(ElectraCorrector, self).__init__()
        self.name = 'electra_corrector'
        t1 = time.time()
        self.g_model = pipeline(
            "fill-mask",
            model=g_model_dir,
            tokenizer=g_model_dir,
            device=device,  # gpu device id
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
                                    details.append((s, token_str, start_idx + idx, start_idx + idx + 1))
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
    d = ErnieCorrector()
    error_sentences = [
        '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',
        '疝気医院那好 ，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.ernie_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
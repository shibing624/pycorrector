# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use ernie detect and correct chinese char error
"""

import operator
import os
import sys
import time

import numpy as np
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L

sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector
from pycorrector.ernie.modeling_ernie import ErnieModelForPretraining, ErnieModel
from pycorrector.ernie.tokenizing_ernie import ErnieTokenizer
from pycorrector.utils.tokenizer import segment

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
        self.model = ErnieCloze.from_pretrained(model_dir)
        self.model.eval()
        logger.debug('Loaded ernie model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))
        self.mask_id = self.ernie_tokenizer.mask_id  # 3
        self.mask_token = self.rev_dict[self.mask_id]  # "[MASK]"
        logger.debug('ernie mask_id :{}, mask_token: {}'.format(self.mask_id, self.mask_token))
        self.topN = topN

    def predict_mask(self, sentence_with_mask):
        ids, id_types = self.ernie_tokenizer.encode(sentence_with_mask)
        ids = np.expand_dims(ids, 0)
        ids = D.to_variable(ids)
        logits = self.model(ids).numpy()
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
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = self.split_text_by_maxlen(text, maxlen=512)
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
                    top_tokens = []
                    for p in predicts:
                        top_tokens.append(p.get('token', ''))

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    d = ErnieCorrector()
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.ernie_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))

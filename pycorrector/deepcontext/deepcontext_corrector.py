# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import operator
import os
import sys
import time
from typing import List

import torch
from loguru import logger

sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.corrector import Corrector
from pycorrector.utils.tokenizer import split_text_into_sentences_by_length
from pycorrector.utils.get_file import get_file
from pycorrector.detector import USER_DATA_DIR
from pycorrector.deepcontext.deepcontext_model import DeepContextModel

pwd_path = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '《', '》']
pretrained_deepcontext_models = {
    # LM model
    'deepcontext_lm.tar.gz':
        'https://github.com/shibing624/pycorrector/releases/download/0.4.5/deepcontext_lm.tar.gz'
}


class DeepContextCorrector(Corrector):
    def __init__(
            self,
            model_name_or_path: str = None,
            max_length: int = 512,
            *args,
            **kwargs,
    ):
        super(DeepContextCorrector, self).__init__(*args, **kwargs)
        if model_name_or_path is not None:
            model_dir = model_name_or_path
        else:
            model_dir = os.path.join(USER_DATA_DIR, 'deepcontext_models', 'deepcontext_lm')
            logger.debug(f'Use default model: {model_dir}')
            filename = 'deepcontext_lm.tar.gz'
            url = pretrained_deepcontext_models.get(filename)
            get_file(
                filename,
                url,
                extract=True,
                cache_dir=USER_DATA_DIR,
                cache_subdir="deepcontext_models",
                verbose=1
            )
        t1 = time.time()
        self.model = DeepContextModel(model_dir=model_dir, max_length=max_length)
        self.max_length = max_length
        logger.debug('Loaded model: %s, spend: %.4f s.' % (model_dir, time.time() - t1))

    def correct(self, sentence: str, **kwargs):
        """Correct the Chinese sentence with deep context language model."""
        details = []
        text_new = ''
        blocks = split_text_into_sentences_by_length(sentence, 128)
        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):
                # 处理中文错误
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    sentence_lst[idx] = self.model.mask
                    # 预测，默认取top10
                    predict_words = self.model.predict_mask_token(sentence_lst, idx, k=10)
                    top_tokens = []
                    for w, _ in predict_words:
                        top_tokens.append(w)

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append((s, token_str, start_idx + idx))
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return {'source': sentence, 'target': text_new, 'errors': details}

    def correct_batch(self, sentences: List[str], **kwargs):
        """
        批量句子纠错
        :param sentences: 句子文本列表
        :param kwargs: 其他参数
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        return [self.correct(s, **kwargs) for s in sentences]

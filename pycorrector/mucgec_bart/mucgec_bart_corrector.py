# -*- coding: utf-8 -*-
import os
import time
from typing import List

import torch
from loguru import logger
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append('../..')
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pycorrector.mucgec_bart.monkey_pack import Pipeline
from pycorrector.utils.sentence_utils import long_sentence_split
import difflib


class MuCGECBartCorrector:
    def __init__(self, model_name_or_path: str = "damo/nlp_bart_text-error-correction_chinese"):
        t1 = time.time()
        self.model = pipeline(Tasks.text_error_correction, model=model_name_or_path)
        logger.debug("Device: {}".format(device))
        logger.debug('Loaded mucgec bart correction model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    def _predict(self, sentences, batch_size=32, max_length=128, silent=True):
        raise NotImplementedError
    
    
    def correct_batch(self, sentences: List[str], max_length: int = 128, batch_size: int = 32, silent: bool = True, ignore_function=None):
        """
        批量句子纠错
        :param sentences: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, bz
        :param silent: bool, show log
        :param ignore_function: function, 自定义一个函数可以指定跳过某类错误， 无需训练模型
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        result = self.model(sentences, batch_size=batch_size, model_name="batch_correct")
        start_idx = 0
        n = len(sentences)
        data = []
        result = [r["output"] for r in result]
        for i in range(n):
            a, b = sentences[i], result[i]
            if len(a)==0 or len(b)==0 or a=="\n":
                start_idx += len(a)
                return
            s = difflib.SequenceMatcher(None, a, b)
            errors = []
            offset = 0
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag!="equal":
                    e = [a[i1:i2], b[j1+offset:j2+offset], i1]
                    if ignore_function and ignore_function(e):
                        # 因为不认为是错误， 所以改回原来的偏移值
                        b = b[:j1] + a[i1:i2] + b[j2:]
                        offset += i2-i1-j2+j1
                        continue
                    
                    errors.append(tuple(e))
            data.append({"source": a, "target": b, "errors": errors})
        return data
        

    def correct(self, sentence: str, **kwargs):
        """长句改为短句, 可直接调用长文本"""
        sentences = long_sentence_split(sentence, max_length=kwargs.pop("max_length", 128), period=kwargs.pop("period", None), comma=kwargs.pop("comma", None))
        batch_results = self.correct_batch(sentences, **kwargs)
        source, target, errors = "", "", []
        for sr in batch_results:
            ll = len(source)
            source += sr["source"]
            target += sr["target"]
            for e in sr["errors"]:
                # 改写位置
                e = list(e)
                e.append(e[-1])
                e[2] += ll
                errors.append(tuple(e))
        return {"source": source, "target": target, "errors": errors, "sentences": batch_results}




    


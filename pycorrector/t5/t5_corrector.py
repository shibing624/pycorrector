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
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.append('../..')
from pycorrector.utils.tokenizer import split_text_into_sentences_by_length
from pycorrector.utils.text_utils import is_chinese

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']


class T5Corrector:
    def __init__(self, model_name_or_path: str = ""):
        self.name = 't5_corrector'
        bin_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_name_or_path = "shibing624/mengzi-t5-base-chinese-correction"
            logger.warning(f'model {bin_path} not exists, use default HF model {model_name_or_path}')
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded t5 correction model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    @staticmethod
    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                continue
            if ori_char in unk_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if ori_char != corrected_text[i]:
                if not is_chinese(ori_char):
                    # pass not chinese char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                if not is_chinese(corrected_text[i]):
                    corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def _predict(self, sentences, batch_size=32, max_length=128, silent=True):
        """Predict sentences with t5 model"""
        corrected_sents = []
        details = []
        for batch in tqdm([sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)],
                          desc="Generating outputs", disable=silent):
            inputs = self.tokenizer(batch, padding=True, max_length=max_length, truncation=True,
                                    return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            for i, sent in enumerate(batch):
                decode_tokens = self.tokenizer.decode(outputs[i], skip_special_tokens=True).replace(' ', '')
                corrected_sent = decode_tokens[:len(sent)]
                corrected_sent, sub_details = self.get_errors(corrected_sent, sent)
                corrected_sents.append(corrected_sent)
                details.append(sub_details)
        return corrected_sents, details

    def correct_batch(self, sentences: List[str], max_length: int = 128, batch_size: int = 32, silent: bool = True):
        """
        批量句子纠错
        :param sentences: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, bz
        :param silent: bool, show log
        :return: list, (corrected_text, [error_word, correct_word, begin_pos, end_pos])
        """
        input_sents = []
        sent_map = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) > max_length:
                # split long sentence into short ones
                short_sentences = [i[0] for i in split_text_into_sentences_by_length(sentence, max_length)]
                input_sents.extend(short_sentences)
                sent_map.extend([idx] * len(short_sentences))
            else:
                input_sents.append(sentence)
                sent_map.append(idx)

        # batch predict
        corrected_sents, details = self._predict(
            input_sents,
            batch_size=batch_size,
            max_length=max_length,
            silent=silent,
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        corrected_details = [[] for _ in range(len(sentences))]
        for idx, corrected_sent, detail in zip(sent_map, corrected_sents, details):
            corrected_sentences[idx] += corrected_sent
            corrected_details[idx].extend(detail)

        return corrected_sentences, corrected_details

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with t5 csc model"""
        corrected_sentences, corrected_details = self.correct_batch([sentence], **kwargs)
        return corrected_sentences[0], corrected_details[0]


if __name__ == "__main__":
    m = T5Corrector()
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
    t2 = time.time()
    res, details = m.correct_batch(error_sentences, batch_size=16, silent=False)
    for sent, corrected_s, d in zip(error_sentences, res, details):
        print("original sentence:{} => {} err:{}".format(sent, corrected_s, d))
    print('[batch]spend time:', time.time() - t2)

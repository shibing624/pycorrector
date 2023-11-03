# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
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
from transformers import BertTokenizerFast, BertForMaskedLM

sys.path.append('../..')
from pycorrector.utils.tokenizer import split_text_into_sentences_by_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤', '\t', '֍', '玕', '']


class MacBertCorrector:
    def __init__(self, model_name_or_path="shibing624/macbert4csc-base-chinese"):
        t1 = time.time()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    @staticmethod
    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                break
            if ori_char in unk_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def _predict(self, sentences, threshold=0.75, batch_size=32, silent=True):
        """Predict sentences with macbert4csc model"""
        corrected_sents = []
        details = []
        for batch in tqdm(
                [
                    sentences[i: i + batch_size]
                    for i in range(0, len(sentences), batch_size)
                ],
                desc="Generating outputs",
                disable=silent,
        ):
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            for id, (logit_tensor, sentence) in enumerate(zip(outputs.logits, batch)):
                decode_tokens_new = self.tokenizer.decode(
                    torch.argmax(logit_tensor, dim=-1), skip_special_tokens=True).split(' ')
                decode_tokens_old = self.tokenizer.decode(
                    inputs['input_ids'][id], skip_special_tokens=True).split(' ')
                decode_tokens_new = decode_tokens_new[:len(decode_tokens_old)]
                probs = torch.max(torch.softmax(logit_tensor, dim=-1), dim=-1)[0].cpu().numpy()
                decode_tokens = ''
                for i in range(len(decode_tokens_old)):
                    if probs[i + 1] >= threshold:
                        decode_tokens += decode_tokens_new[i]
                    else:
                        decode_tokens += decode_tokens_old[i]
                corrected_text = decode_tokens[:len(sentence)]
                corrected_text, sub_details = self.get_errors(corrected_text, sentence)
                corrected_sents.append(corrected_text)
                details.append(sub_details)
        return corrected_sents, details

    def correct_batch(
            self,
            sentences: List[str],
            max_length: int = 128,
            batch_size: int = 32,
            threshold: float = 0.7,
            silent: bool = True
    ):
        """
        Correct sentences with macbert4csc model
        :param sentences: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, batch size
        :param threshold: float, threshold of error word
        :param silent: bool, silent or not
        :return: corrected_sentences, corrected_details, (List[str], [[error_word, correct_word, begin_pos, end_pos], ...])
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

        # predict all sentences
        sents, details = self._predict(
            input_sents,
            threshold=threshold,
            batch_size=batch_size,
            silent=silent,
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        corrected_details = [[] for _ in range(len(sentences))]
        for idx, corrected_sent, detail in zip(sent_map, sents, details):
            corrected_sentences[idx] += corrected_sent
            corrected_details[idx].extend(detail)

        return corrected_sentences, corrected_details

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with macbert4csc model"""
        corrected_sentences, corrected_details = self.correct_batch([sentence], **kwargs)
        return corrected_sentences[0], corrected_details[0]


if __name__ == "__main__":
    m = MacBertCorrector()
    error_sentences = [
        '内容提要——在知识产权学科领域里',
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
    ]
    t2 = time.time()
    res, details = m.correct_batch(error_sentences, threshold=0.75, batch_size=16, silent=False)
    for sent, corrected_s, d in zip(error_sentences, res, details):
        print("original sentence:{} => {} err:{}".format(sent, corrected_s, d))
    print('[batch]spend time:', time.time() - t2)

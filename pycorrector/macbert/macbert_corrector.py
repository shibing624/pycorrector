# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
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
from pycorrector.utils.error_utils import get_errors

device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MacBertCorrector:
    def __init__(self, model_name_or_path="shibing624/macbert4csc-base-chinese"):
        t1 = time.time()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    def _predict(self, sentences, threshold=0.7, batch_size=32, silent=True):
        """Predict sentences with macbert4csc model"""
        corrected_sents = []
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
                decode_tokens_new = decode_tokens_new[:len(sentence)]
                if len(decode_tokens_new) == len(sentence):
                    probs = torch.max(torch.softmax(logit_tensor, dim=-1), dim=-1)[0].cpu().numpy()
                    decode_str = ''
                    for i in range(len(sentence)):
                        if probs[i + 1] >= threshold:
                            decode_str += decode_tokens_new[i]
                        else:
                            decode_str += sentence[i]
                    corrected_text = decode_str
                else:
                    corrected_text = sentence
                corrected_sents.append(corrected_text)
        return corrected_sents

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
        :param threshold: float, threshold of error word,
            阈值越大（如 0.9），模型对预测的置信度要求越高，只有当模型非常确信某个字需要被纠正时才会执行替换，这会减少误纠正但可能会漏掉一些实际错误;
            阈值越小（如 0.5），模型对预测的置信度要求越低，更容易对文本进行修改，可能会纠正更多错误但也可能引入不必要的修改
        :param silent: bool, silent or not
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
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
        sents = self._predict(
            input_sents,
            threshold=threshold,
            batch_size=batch_size,
            silent=silent,
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, sents):
            corrected_sentences[idx] += corrected_sent

        new_corrected_sentences = []
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            new_corrected_sent, sub_details = get_errors(corrected_sent, sentences[idx])
            new_corrected_sentences.append(new_corrected_sent)
            corrected_details.append(sub_details)
        return [{'source': s, 'target': c, 'errors': e} for s, c, e in
                zip(sentences, new_corrected_sentences, corrected_details)]

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with macbert4csc model"""
        return self.correct_batch([sentence], **kwargs)[0]

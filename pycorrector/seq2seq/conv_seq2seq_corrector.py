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

from pycorrector.seq2seq.conv_seq2seq_model import ConvSeq2SeqModel

from pycorrector.utils.tokenizer import split_text_into_sentences_by_length
from pycorrector.utils.get_file import get_file
from pycorrector.detector import USER_DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '《', '》']
pre_trained_seq2seq_models = {
    # ConvSeq2Seq model 4.6MB
    'convseq2seq_correction.tar.gz':
        'https://github.com/shibing624/pycorrector/releases/download/0.4.5/convseq2seq_correction.tar.gz'
}


class ConvSeq2SeqCorrector:
    def __init__(
            self,
            model_name_or_path: str = '',
            max_length: int = 128,
    ):
        bin_path = os.path.join(model_name_or_path, 'convseq2seq.pth')
        if model_name_or_path and os.path.exists(bin_path):
            model_dir = model_name_or_path
        else:
            model_dir = os.path.join(USER_DATA_DIR, 'seq2seq_models', 'convseq2seq_correction')
            logger.debug(f'model {bin_path} not exists, use default model: {model_dir}')
            filename = 'convseq2seq_correction.tar.gz'
            url = pre_trained_seq2seq_models.get(filename)
            get_file(
                filename,
                url,
                extract=True,
                cache_dir=USER_DATA_DIR,
                cache_subdir="seq2seq_models",
                verbose=1
            )
        t1 = time.time()
        logger.debug("Device: {}".format(device))
        self.model = ConvSeq2SeqModel(model_dir=model_dir)
        self.max_length = max_length
        logger.debug('Loaded model: %s, spend: %.4f s.' % (model_dir, time.time() - t1))

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
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def correct_batch(self, sentences: List[str], max_length: int = 128, silent: bool = True):
        """
        批量句子纠错
        :param: sentences, List[str]: 待纠错的句子
        :param: max_length, int: 句子最大长度
        :param: silent, bool: 是否打印日志
        :return: list, [corrected_texts, [error_word, correct_word, begin_pos, end_pos]]
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
        corrected_sents = self.model.predict(input_sents, silent=silent)

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, corrected_sents):
            corrected_sentences[idx] += corrected_sent
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            _, sub_details = self.get_errors(corrected_sent, sentences[idx])
            corrected_details.append(sub_details)

        return corrected_sentences, corrected_details

    def correct(self, sentence: str):
        """Correct a sentence with conv seq2seq model"""
        corrected_sentences, corrected_details = self.correct_batch([sentence])
        return corrected_sentences[0], corrected_details[0]


if __name__ == "__main__":
    m = ConvSeq2SeqCorrector()
    error_sentences = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    res, ds = m.correct_batch(error_sentences, silent=False)
    for sent, r, d in zip(error_sentences, res, ds):
        print("original sentence:{} => {} , err:{}".format(sent, r, d))

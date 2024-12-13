# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
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
from pycorrector.utils.error_utils import get_errors

device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
pretrained_seq2seq_models = {
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
            url = pretrained_seq2seq_models.get(filename)
            get_file(
                filename,
                url,
                extract=True,
                cache_dir=USER_DATA_DIR,
                cache_subdir="seq2seq_models",
                verbose=1
            )
        t1 = time.time()
        self.model = ConvSeq2SeqModel(model_dir=model_dir)
        self.max_length = max_length
        logger.debug('Loaded model: %s, spend: %.4f s.' % (model_dir, time.time() - t1))

    def correct_batch(self, sentences: List[str], max_length: int = 128, silent: bool = True):
        """
        批量句子纠错
        :param: sentences, List[str]: 待纠错的句子
        :param: max_length, int: 句子最大长度
        :param: silent, bool: 是否打印日志
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

        # batch predict
        corrected_sents = self.model.predict(input_sents, silent=silent)

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, corrected_sents):
            corrected_sentences[idx] += corrected_sent
        new_corrected_sentences = []
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            new_corrected_sent, sub_details = get_errors(corrected_sent, sentences[idx])
            new_corrected_sentences.append(new_corrected_sent)
            corrected_details.append(sub_details)
        return [{'source': s, 'target': c, 'errors': e} for s, c, e in
                zip(sentences, new_corrected_sentences, corrected_details)]

    def correct(self, sentence: str):
        """Correct a sentence with conv seq2seq model"""
        return self.correct_batch([sentence])[0]

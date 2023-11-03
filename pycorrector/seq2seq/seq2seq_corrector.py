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

import numpy as np
import torch
from loguru import logger

sys.path.append('../..')

from pycorrector.seq2seq.data_reader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_word_dict
from pycorrector.seq2seq.conv_seq2seq import ConvSeq2Seq

from pycorrector.utils.get_file import get_file
from pycorrector.detector import USER_DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '《', '》']
pre_trained_seq2seq_models = {
    # ConvSeq2Seq model 4.6MB
    'convseq2seq_correction.tar.gz':
        'https://github.com/shibing624/pycorrector/releases/download/0.4.5/convseq2seq_correction.tar.gz'
}


class Seq2SeqCorrector:
    def __init__(
            self,
            model_name_or_path: str = '',
            embed_size: int = 128,
            hidden_size: int = 128,
            dropout: float = 0.25,
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
        src_vocab_path = os.path.join(model_dir, 'vocab_source.txt')
        trg_vocab_path = os.path.join(model_dir, 'vocab_target.txt')
        self.src_2_ids = load_word_dict(src_vocab_path)
        self.trg_2_ids = load_word_dict(trg_vocab_path)
        self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
        self.model = ConvSeq2Seq(
            encoder_vocab_size=len(self.src_2_ids),
            decoder_vocab_size=len(self.trg_2_ids),
            embed_size=embed_size,
            enc_hidden_size=hidden_size,
            dec_hidden_size=hidden_size,
            dropout=dropout,
            trg_pad_idx=trg_pad_idx,
            device=device,
            max_length=max_length
        ).to(device)
        model_path = os.path.join(model_dir, 'convseq2seq.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.max_length = max_length
        logger.debug('Loaded model: %s, spend: %.4f s.' % (model_path, time.time() - t1))

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

    def predict(self, sentences: List[str]):
        corrected_sents = []
        details = []
        for query in sentences:
            out = []
            tokens = [token.lower() for token in query]
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]

            sos_idx = self.trg_2_ids[SOS_TOKEN]
            src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
            translation, attn = self.model.translate(src_tensor, sos_idx)
            translation = [self.id_2_trgs[i] for i in translation if i in self.id_2_trgs]
            for word in translation:
                if word != EOS_TOKEN:
                    out.append(word)
                else:
                    break
            corrected_sent = ''.join(out)
            corrected_sent, sub_details = self.get_errors(corrected_sent, query)
            corrected_sents.append(corrected_sent)
            details.append(sub_details)
        return corrected_sents, details

    def correct_batch(self, sentences: List[str]):
        """
        批量句子纠错
        :param: sentences, List[str]: 待纠错的句子
        :return: list, [corrected_texts, [error_word, correct_word, begin_pos, end_pos]]
        """
        return self.predict(sentences)

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with conv seq2seq model"""
        corrected_sentences, corrected_details = self.correct_batch([sentence])
        return corrected_sentences[0], corrected_details[0]


if __name__ == "__main__":
    m = Seq2SeqCorrector()
    error_sentences = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    res, ds = m.correct_batch(error_sentences)
    for sent, r, d in zip(error_sentences, res, ds):
        print("original sentence:{} => {} , err:{}".format(sent, r, d))

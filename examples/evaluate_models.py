# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys

sys.path.append("..")

import pycorrector
from pycorrector.utils import eval

pwd_path = os.path.abspath(os.path.dirname(__file__))


def main(args):
    if args.data == 'sighan_15' and args.model == 'kenlm':
        # Sentence Level: acc:0.5100, precision:0.5139, recall:0.1363, f1:0.2154, cost time:1464.87 s
        eval.eval_sighan2015_by_model(pycorrector.correct)
    if args.data == 'sighan_15' and args.model == 'macbert':
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_sighan2015_by_model_batch(model.batch_macbert_correct)
        # macbert-base: Sentence Level: acc:0.7900, precision:0.8250, recall:0.7293, f1:0.7742, cost time:4.90 s
        # pert-base:    Sentence Level: acc:0.7709, precision:0.7893, recall:0.7311, f1:0.7591, cost time:2.52 s, total num: 1100
        # pert-large:   Sentence Level: acc:0.7709, precision:0.7847, recall:0.7385, f1:0.7609, cost time:7.22 s, total num: 1100
        eval.eval_sighan2015_by_model(model.macbert_correct)
    if args.data == 'sighan_15' and args.model == 'bartseq2seq':
        from transformers import BertTokenizerFast
        from textgen import BartSeq2SeqModel
        tokenizer = BertTokenizerFast.from_pretrained('shibing624/bart4csc-base-chinese')
        model = BartSeq2SeqModel(
            encoder_type='bart',
            encoder_decoder_type='bart',
            encoder_decoder_name='shibing624/bart4csc-base-chinese',
            tokenizer=tokenizer,
            args={"max_length": 128})
        eval.eval_sighan2015_by_model_batch(model.predict)
        # Sentence Level: acc:0.6845, precision:0.6984, recall:0.6354, f1:0.6654

    if args.data == 'corpus500' and args.model == 'kenlm':
        # right_rate:0.486, right_count:243, total_count:500;
        # recall_rate:0.18, recall_right_count:54, recall_total_count:300, spend_time:78 s
        eval.eval_corpus500_by_model(pycorrector.correct)
    if args.data == 'corpus500' and args.model == 'macbert':
        # Sentence Level: acc:0.724000, precision:0.912821, recall:0.595318, f1:0.720648, cost time:6.43 s
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_corpus500_by_model(model.macbert_correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sighan_15', help='evaluate dataset, sighan_15/corpus500')
    parser.add_argument('--model', type=str, default='kenlm', help='which model to evaluate, kenlm/bert/macbert/ernie')
    args = parser.parse_args()
    main(args)

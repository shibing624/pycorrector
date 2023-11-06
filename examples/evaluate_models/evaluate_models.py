# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append("../..")

import pycorrector
from pycorrector.utils import eval


def main(args):
    if args.model == 'kenlm':
        # Sentence Level: acc:0.5100, precision:0.5139, recall:0.1363, f1:0.2154, cost time:1464.87 s
        eval.eval_sighan2015_by_model(pycorrector.correct)
    elif args.model == 'macbert':
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_sighan2015_by_model_batch(model.correct)
        # macbert-base: Sentence Level: acc:0.7900, precision:0.8250, recall:0.7293, f1:0.7742, cost time:4.90 s
        # pert-base:    Sentence Level: acc:0.7709, precision:0.7893, recall:0.7311, f1:0.7591, cost time:2.52 s, total num: 1100
        # pert-large:   Sentence Level: acc:0.7709, precision:0.7847, recall:0.7385, f1:0.7609, cost time:7.22 s, total num: 1100
    elif args.model == 'bartseq2seq':
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
    elif args.model == 'chatglm':
        from pycorrector.gpt.gpt_corrector import GptCorrector
        model = GptCorrector()
        eval.eval_sighan2015_by_model_batch(model.correct_batch)
        # chatglm3-6b-csc: Sentence Level: acc:
    else:
        raise ValueError('model name error.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kenlm', help='which model to evaluate, kenlm/bert/macbert/ernie')
    args = parser.parse_args()
    main(args)

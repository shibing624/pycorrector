# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append("../..")

from pycorrector import eval_sighan2015_by_model_batch


def main(args):
    if args.model == 'kenlm':
        from pycorrector import Corrector
        m = Corrector()
        eval_sighan2015_by_model_batch(m.correct_batch)
        # Sentence Level: acc:0.5409, precision:0.6532, recall:0.1492, f1:0.2429, cost time:295.07 s, total num: 1100
    elif args.model == 'macbert':
        from pycorrector import MacBertCorrector
        model = MacBertCorrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # macbert:      Sentence Level: acc:0.7891, precision:0.8479, recall:0.6980, f1:0.7657, cost time:2.37 s, total num: 1100
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
        eval_sighan2015_by_model_batch(model.predict)
        # Sentence Level: acc:0.6845, precision:0.6984, recall:0.6354, f1:0.6654
    elif args.model == 'seq2seq':
        from pycorrector import ConvSeq2SeqCorrector
        model = ConvSeq2SeqCorrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # Sentence Level: acc:0.3909, precision:0.2803, recall:0.1492, f1:0.1947, cost time:219.50 s, total num: 1100
    elif args.model == 't5':
        from pycorrector import T5Corrector
        model = T5Corrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # Sentence Level: acc:0.6291, precision:0.8199, recall:0.3186, f1:0.4589, cost time:27.54 s, total num: 1100
    elif args.model == 'deepcontext':
        from pycorrector import DeepContextCorrector
        model = DeepContextCorrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # Sentence Level: acc:
    elif args.model == 'ernie_csc':
        from pycorrector import ErnieCscCorrector
        model = ErnieCscCorrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # Sentence Level: acc:0.7491, precision:0.7623, recall:0.7145, f1:0.7376, cost time:3.03 s, total num: 1100
    elif args.model == 'chatglm':
        from pycorrector.gpt.gpt_corrector import GptCorrector
        model = GptCorrector()
        eval_sighan2015_by_model_batch(model.correct_batch)
        # chatglm3-6b-csc: Sentence Level: acc:0.5564, precision:0.5574, recall:0.4917, f1:0.5225, cost time:1572.49 s, total num: 1100
    else:
        raise ValueError('model name error.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kenlm', help='which model to evaluate')
    args = parser.parse_args()
    main(args)

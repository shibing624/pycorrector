# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
import os

sys.path.append("../..")

from pycorrector import eval_model_batch

pwd_path = os.path.abspath(os.path.dirname(__file__))


def main(args):
    if args.model == 'kenlm':
        from pycorrector import Corrector
        m = Corrector()
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch)
            # Sentence Level: acc:0.5409, precision:0.6532, recall:0.1492, f1:0.2429, cost time:295.07 s, total num: 1100
            # Sentence Level: acc:0.5502, precision:0.8022, recall:0.1957, f1:0.3147, cost time:37.28 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.5790, precision:0.8581, recall:0.2410, f1:0.3763, cost time:64.61 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.5850, precision:0.7518, recall:0.2128, f1:0.3317, cost time:30.61 s, total num: 1000
    elif args.model == 'macbert':
        from pycorrector import MacBertCorrector
        model = MacBertCorrector()
        if args.data == 'sighan':
            eval_model_batch(model.correct_batch)
            # macbert:      Sentence Level: acc:0.7918, precision:0.8489, recall:0.7035, f1:0.7694, cost time:2.25 s, total num: 1100
            # pert-base:    Sentence Level: acc:0.7709, precision:0.7893, recall:0.7311, f1:0.7591, cost time:2.52 s, total num: 1100
            # pert-large:   Sentence Level: acc:0.7709, precision:0.7847, recall:0.7385, f1:0.7609, cost time:7.22 s, total num: 1100
            # macbert4csc   Sentence Level: acc:0.8388, precision:0.9274, recall:0.7534, f1:0.8314, cost time:4.26 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(model.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.2390, precision:0.1921, recall:0.1385, f1:0.1610, cost time:7.11 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(model.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.5360, precision:0.6000, recall:0.1240, f1:0.2055, cost time:2.65 s, total num: 1000
    elif args.model == 'seq2seq':
        from pycorrector import ConvSeq2SeqCorrector
        model = ConvSeq2SeqCorrector()
        eval_model_batch(model.correct_batch)
        # Sentence Level: acc:0.3909, precision:0.2803, recall:0.1492, f1:0.1947, cost time:219.50 s, total num: 1100
    elif args.model == 't5':
        from pycorrector import T5Corrector
        m = T5Corrector()
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch)
            # Sentence Level: acc:0.7582, precision:0.8321, recall:0.6390, f1:0.7229, cost time:26.36 s, total num: 1100
            # Sentence Level: acc:0.7907, precision:0.8920, recall:0.6863, f1:0.7758, cost time:20.82 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.5230, precision:0.6471, recall:0.2087, f1:0.3156, cost time:43.61 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.4650, precision:0.2743, recall:0.0640, f1:0.1039, cost time:14.99 s, total num: 1000
    elif args.model == 'deepcontext':
        from pycorrector import DeepContextCorrector
        model = DeepContextCorrector()
        eval_model_batch(model.correct_batch)
    elif args.model == 'ernie_csc':
        from pycorrector import ErnieCscCorrector
        m = ErnieCscCorrector()
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch)
            # Sentence Level: acc:0.7491, precision:0.7623, recall:0.7145, f1:0.7376, cost time:3.03 s, total num: 1100
            # Sentence Level: acc:0.8373, precision:0.8817, recall:0.7989, f1:0.8383, cost time:14.97 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.5370, precision:0.6882, recall:0.2220, f1:0.3357, cost time:25.15 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.4600, precision:0.2971, recall:0.0847, f1:0.1318, cost time:18.69 s, total num: 1000
    elif args.model == 'chatglm':
        from pycorrector.gpt.gpt_corrector import GptCorrector
        m = GptCorrector(model_name_or_path="THUDM/chatglm3-6b",
                         model_type='chatglm',
                         peft_name="shibing624/chatglm3-6b-csc-chinese-lora")
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch, prefix_prompt="对这个句子语法纠错\n\n", prompt_template_name='vicuna')
            # Sentence Level: acc:0.5564, precision:0.5574, recall:0.4917, f1:0.5225, cost time:1572.49 s, total num: 1100
            #
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"),
                             prefix_prompt="对这个句子语法纠错\n\n",
                             prompt_template_name='vicuna')
            #
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"),
                             prefix_prompt="对这个句子语法纠错\n\n",
                             prompt_template_name='vicuna')
            #
    elif args.model == 'qwen1.5b':
        from pycorrector.gpt.gpt_corrector import GptCorrector
        m = GptCorrector(model_name_or_path="shibing624/chinese-text-correction-1.5b")
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch)
            # Sentence Level: acc:0.4540, precision:0.4641, recall:0.2252, f1:0.3032, cost time:243.50 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.7990, precision:0.9015, recall:0.6945, f1:0.7846, cost time:266.26 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.9560, precision:0.9889, recall:0.9194, f1:0.9529, cost time:210.11 s, total num: 1000
    elif args.model == 'qwen7b':
        from pycorrector.gpt.gpt_corrector import GptCorrector
        m = GptCorrector(model_name_or_path="shibing624/chinese-text-correction-7b")
        if args.data == 'sighan':
            eval_model_batch(m.correct_batch)
            # Sentence Level: acc:0.5672, precision:0.6463, recall:0.3968, f1:0.4917, cost time:392.10 s, total num: 707
        elif args.data == 'ec_law':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/ec_law_test.tsv"))
            # Sentence Level: acc:0.9790, precision:0.9941, recall:0.9658, f1:0.9798, cost time:717.37 s, total num: 1000
        elif args.data == 'mcsc':
            eval_model_batch(m.correct_batch, input_tsv_file=os.path.join(pwd_path, "../data/mcsc_test.tsv"))
            # Sentence Level: acc:0.9960, precision:0.9979, recall:0.9938, f1:0.9959, cost time:267.12 s, total num: 1000
    else:
        raise ValueError('model name error.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='macbert', help='which model to evaluate')
    parser.add_argument('--data', type=str, default='sighan', help='test dataset')
    args = parser.parse_args()
    main(args)

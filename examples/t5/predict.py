# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
sys.path.append('../..')

from pycorrector.t5.t5_corrector import T5Corrector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs-mengzi-t5-base-chinese-correction-v1/')
    args = parser.parse_args()
    print(args)
    model = T5Corrector(args.output_dir)
    example_sentences = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    corrected_sents = model.correct_batch(example_sentences)
    for i, o in zip(example_sentences, corrected_sents):
        print(i, ' -> ', o)


if __name__ == '__main__':
    main()

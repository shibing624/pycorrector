# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='output/mengzi-t5-base-chinese-correction/', help='save dir')
    args = parser.parse_args()
    print(args)
    model_dir = args.save_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    example_sentences = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    corrected_sents = []
    for s in example_sentences:
        model_inputs = tokenizer(s, max_length=128, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**model_inputs, max_length=128)
        r = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_sents.append(r)
    for i, o in zip(example_sentences, corrected_sents):
        print(i, ' -> ', o)


if __name__ == '__main__':
    main()

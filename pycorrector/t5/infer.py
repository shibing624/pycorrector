# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='output', help='save dir')
    args = parser.parse_args()
    return args


def predict():
    example_sentences = ["我跟我朋唷打算去法国玩儿。",
                         "少先队员因该为老人让坐。",
                         "我们是新时代的接斑人",
                         "我咪路，你能给我指路吗？",
                         "他带了黑色的包，也带了照像机",
                         ]
    args = parse_args()
    model_dir = os.path.join(args.save_dir, './byt5-base-zh-correction')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    results = []
    for s in example_sentences:
        model_inputs = tokenizer(s, max_length=128, truncation=True, return_tensors="pt")
        outputs = model.generate(**model_inputs, max_length=128)
        r = tokenizer.decode(outputs[0])
        print('output:', r)
        results.append(r)
    return results


if __name__ == '__main__':
    predict()

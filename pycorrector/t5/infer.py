# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='output/mengzi-t5-base-chinese-correction/', help='save dir')
    args = parser.parse_args()
    return args


def predict():
    example_sentences = [
        "我跟我朋唷打算去法国玩儿。",
        "少先队员因该为老人让坐。",
        "我们是新时代的接斑人",
        "我咪路，你能给我指路吗？",
        "他带了黑色的包，也带了照像机",
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '妈妈说："别趴地上了，快起来，你还吃饭吗？"，我说："好。"就扒起来了。',
        '你说：“怎么办？”我怎么知道？',
        '我父母们常常说：“那时候吃的东西太少，每天只能吃一顿饭。”想一想，人们都快要饿死，谁提出化肥和农药的污染。',
        '这本新书《居里夫人传》将的很生动有趣',
        '֍我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……֍新的食谱',
        '注意：“跨类保护”不等于“全类保护”。',
        '12.——对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性；',
        '《著作权法》（2020修正）第23条：“自然人的作品，其发表权、本法第',
        '三步检验法（三步检验标准）（three-step test）：若要',
        '三步检验法“三步‘检验’标准”（three-step test）：若要',
    ]
    args = parse_args()
    model_dir = args.save_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    results = []
    for s in example_sentences:
        model_inputs = tokenizer(s, max_length=128, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**model_inputs, max_length=128)
        r = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('output:', r)
        results.append(r)
    return results


if __name__ == '__main__':
    predict()

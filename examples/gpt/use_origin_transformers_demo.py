# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint = "shibing624/chinese-text-correction-1.5b"

# use cuda or mps or cpu
device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

sents = ['少先队员因该为老人让坐。',
         '下个星期，我跟我朋唷打算去法国玩儿。',
         '“明德慎罚”是殷商初期“天命”、“天罚”思想的继承和发扬']


def batch_predict(sents):
    tokenizer.padding_side = "left"  # batch predict must set before tokenization
    conversation = []
    for s in sents:
        messages = [
            {'role': 'system', 'content': '你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。'},
            {'role': 'user', 'content': s}]
        conversation.append(messages)
    inputs = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
        padding=True,
    )
    outputs = model.generate(inputs.to(device), max_new_tokens=512, temperature=0.1)
    for i, o in enumerate(outputs):
        print(f"Input {i}: {sents[i]}")
        print(tokenizer.decode(o, skip_special_tokens=True))
        print('===')


def single_predict(sentence):
    messages = [{'role': 'system', 'content': '你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。'},
                {'role': 'user', 'content': sentence}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    outputs = model.generate(input_text.to(device), max_new_tokens=512, temperature=0.1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    batch_predict(sents)
    for i in sents:
        single_predict(i)

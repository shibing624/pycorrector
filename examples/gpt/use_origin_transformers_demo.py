# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
checkpoint = "shibing624/chinese-text-correction-1.5b"

# use cuda or mps or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

sents = ['文本纠错\n\n少先队员因该为老人让坐。',
         '文本纠错\n\n下个星期，我跟我朋唷打算去法国玩儿。']

for q in sents:
    messages = [{"role": "user", "content": q}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    outputs = model.generate(input_text.to(device), max_new_tokens=512)
    print(tokenizer.decode(outputs[0]))
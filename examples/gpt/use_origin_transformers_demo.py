# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model = PeftModel.from_pretrained(model, "shibing624/chatglm3-6b-csc-chinese-lora")

sents = ['对下面文本纠错\n\n少先队员因该为老人让坐。',
         '对下面文本纠错\n\n下个星期，我跟我朋唷打算去法国玩儿。']


def get_prompt(user_query):
    vicuna_prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. " \
                    "USER: {query} ASSISTANT:"
    return vicuna_prompt.format(query=user_query)


for s in sents:
    q = get_prompt(s)
    input_ids = tokenizer(q).input_ids
    generation_kwargs = dict(max_new_tokens=128, do_sample=True, temperature=0.8)
    outputs = model.generate(input_ids=torch.as_tensor([input_ids]).to('cuda'), **generation_kwargs)
    output_tensor = outputs[0][len(input_ids):]
    response = tokenizer.decode(output_tensor, skip_special_tokens=True)
    print(response)

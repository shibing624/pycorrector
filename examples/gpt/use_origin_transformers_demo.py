# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

from peft import PeftModel
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(model, "shibing624/chatglm3-6b-csc-chinese-lora")
model.half().cuda()  # fp16
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

sents = ['对下面文本纠错\n\n少先队员因该为老人让坐。',
         '对下面文本纠错\n\n下个星期，我跟我朋唷打算去法国玩儿。']
for s in sents:
    response = model.chat(tokenizer, s, max_length=128, eos_token_id=tokenizer.eos_token_id)
    print(response)

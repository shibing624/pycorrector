# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

import os

import torch
from transformers import ElectraForPreTraining, ElectraTokenizer, pipeline

pwd_path = os.path.abspath(os.path.dirname(__file__))

D_model_dir = os.path.join(pwd_path, "../data/electra_models/chinese_electra_base_discriminator_pytorch/")
G_model_dir = os.path.join(pwd_path, "../data/electra_models/chinese_electra_base_generator_pytorch/")


def fill_mask_demo():
    nlp = pipeline(
        "fill-mask",
        model=G_model_dir,
        tokenizer=G_model_dir
    )

    print(
        nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks.")
    )

    i = nlp('hi lili, What is the name of the [MASK] ?')
    print(i)

    i = nlp('今天[MASK]情很好')
    print(i)


def detect_error_demo():
    tokenizer = ElectraTokenizer.from_pretrained(D_model_dir)
    discriminator = ElectraForPreTraining.from_pretrained(D_model_dir)

    sentence = '今天新情很好'
    fake_tokens = tokenizer.tokenize(sentence)
    print(fake_tokens)
    fake_inputs = tokenizer.encode(sentence, return_tensors="pt")

    discriminator_outputs = discriminator(fake_inputs)
    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    print(list(zip(fake_tokens, predictions.tolist())))
    print("fixed " + '*' * 42)
    print(list(zip(fake_tokens, predictions.tolist()[1:-1])))


if __name__ == '__main__':
    detect_error_demo()
    fill_mask_demo()

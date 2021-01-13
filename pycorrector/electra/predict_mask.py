# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

import torch

sys.path.append('../..')
from pycorrector.transformers import pipeline, ElectraForPreTraining, ElectraTokenizer
from pycorrector import config


def fill_mask_demo():
    nlp = pipeline(
        "fill-mask",
        model=config.electra_G_model_dir,
        tokenizer=config.electra_G_model_dir,
        device=0,  # gpu device id
    )
    print(nlp.tokenizer.mask_token)
    print(
        nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks.")
    )

    i = nlp('hi, What is your [MASK] ?')
    print(i)

    i = nlp('今天[MASK]情很好')
    print(i)


def detect_error_demo():
    tokenizer = ElectraTokenizer.from_pretrained(config.electra_D_model_dir)
    discriminator = ElectraForPreTraining.from_pretrained(config.electra_D_model_dir)

    sentence = '今天新情很好'
    fake_tokens = tokenizer.tokenize(sentence)
    print(fake_tokens)
    fake_inputs = tokenizer.encode(sentence, return_tensors="pt")

    discriminator_outputs = discriminator(fake_inputs)
    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    print(list(zip(fake_tokens, predictions.tolist())))
    print("fixed " + '*' * 42)
    print(predictions.tolist())
    print(list(zip(fake_tokens, predictions.tolist()[0][1:-1])))


if __name__ == '__main__':
    detect_error_demo()
    fill_mask_demo()

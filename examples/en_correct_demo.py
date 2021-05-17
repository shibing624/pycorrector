# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("..")

import pycorrector

if __name__ == '__main__':
    sent = "what happending? how to speling it can you gorrect it."
    r = pycorrector.en_correct(sent)
    print(sent, '=>', r)

    sent_lst = ['what', 'hapenning', 'how', 'to', 'speling', 'it', 'you', 'can', 'gorrect', 'it']
    for i in sent_lst:
        print(i, '=>', pycorrector.en_correct(i))

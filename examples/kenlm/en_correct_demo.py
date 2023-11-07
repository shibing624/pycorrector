# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
sys.path.append("../..")
from pycorrector import EnSpellCorrector

if __name__ == '__main__':
    # 1. 演示英文句子纠错
    sent = "what happending? how to speling it, can you gorrect it?"
    m = EnSpellCorrector()
    details = m.correct(sent)
    print(details)
    print()

    # 2. 演示英文句子列表纠错
    sent_lst = ['what hapenning?','how to speling it', 'gorrect', 'i know']
    for sent in sent_lst:
        details = m.correct(sent)
        print('[error] ', details)
    print()

    # 3. 演示自定义英文词典
    sent = "what is your name? shylock?"
    r = m.correct(sent)
    print(r)
    print('-' * 42)
    my_dict = {'your': 120, 'name': 2, 'is': 1, 'shylock': 1, 'what': 1}  # word, freq
    spell = EnSpellCorrector(word_freq_dict=my_dict)
    r = spell.correct(sent)
    print(r)
    print()

    # 4. 演示自定义纠错集
    spell = EnSpellCorrector()
    sent = "what happt ? who is shylock."
    r = spell.correct(sent)
    print(r)
    print('-' * 42)
    spell.set_en_custom_confusion_dict('./my_custom_confusion.txt')
    r = spell.correct(sent)
    print(r)

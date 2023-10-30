# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
sys.path.append("../..")

import pycorrector

if __name__ == '__main__':
    # 1. 演示英文句子纠错
    sent = "what happending? how to speling it, can you gorrect it?"
    corrected_text, details = pycorrector.en_correct(sent)
    print(sent, '=>', corrected_text)
    print(details)
    print()

    # 2. 演示英文句子列表纠错
    sent_lst = ['what hapenning?','how to speling it', 'gorrect', 'i know']
    for sent in sent_lst:
        corrected_text, details = pycorrector.en_correct(sent)
        if details:
            print('[error] ', sent, '=>', corrected_text, details)
    print()

    # 3. 演示自定义英文词典
    from pycorrector.en_spell import EnSpell

    sent = "what is your name? shylock?"
    spell = EnSpell()
    corrected_text, details = spell.correct(sent)
    print(sent, '=>', corrected_text, details)
    print('-' * 42)
    my_dict = {'your': 120, 'name': 2, 'is': 1, 'shylock': 1, 'what': 1}  # word, freq
    spell = EnSpell(word_freq_dict=my_dict)
    corrected_text, details = spell.correct(sent)
    print(sent, '=>', corrected_text, details)
    print()

    # 4. 演示自定义纠错集
    from pycorrector.en_spell import EnSpell

    spell = EnSpell()
    sent = "what happt ? who is shylock."
    corrected_text, details = spell.correct(sent)
    print(sent, '=>', corrected_text, details)
    print('-' * 42)
    spell.set_en_custom_confusion_dict('./my_custom_confusion.txt')
    corrected_text, details = spell.correct(sent)
    print(sent, '=>', corrected_text, details)

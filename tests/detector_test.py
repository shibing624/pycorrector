# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys
sys.path.append("../")
import pycorrector
from pycorrector.tokenizer import segment

sent1 = '少先队员应该为老人让座'
sent_seg = segment(sent1)
ppl = pycorrector.ppl_score(sent_seg)
print('ppl_score:', ppl)

sent2 = '少先队员因该为老人让坐'
sent_seg = segment(sent2)
ppl = pycorrector.ppl_score(sent_seg)
print('ppl_score:', ppl)

print(sent1, pycorrector.detect(sent1))
print(sent2, pycorrector.detect(sent2))

c = pycorrector.get_same_pinyin('长')
print('get_same_pinyin:', c)

c = pycorrector.get_same_stroke('长')
print('get_same_stroke:', c)

freq = pycorrector.word_frequency('龟龙麟凤')
print('freq:', freq)

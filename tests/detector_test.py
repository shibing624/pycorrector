# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from pycorrector.utils.text_utils import *
import pycorrector

c = pycorrector.get_same_pinyin('长')
print('get_same_pinyin:', c)

c = pycorrector.get_same_stroke('长')
print('get_same_stroke:', c)

freq = pycorrector.word_frequency('龟龙麟凤')
print('freq:', freq)

sent = '少先队员应该为老人让座'
sent_seg = segment(sent)
ppl = pycorrector.ppl_score(sent_seg)
print('ppl_score:', ppl)

sent = '少先队员因该为老人让坐'
sent_seg = segment(sent)
ppl = pycorrector.ppl_score(sent_seg)
print('ppl_score:', ppl)

print(pycorrector.detect(sent))

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

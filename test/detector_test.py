# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

from pycorrector.corrector import *
import pycorrector.detector
from pycorrector.util import segment
c = get_same_pinyin('长')
print('get_same_pinyin:', c)

c = get_same_stroke('长')
print('get_same_stroke:', c)

freq = pycorrector.detector.get_frequency('龟龙麟凤')
print('freq:', freq)

sent = '少先队员应该为老人让座'
sent_seg = segment(sent)
ppl = get_ppl_score(sent_seg)
print('get_ppl_score:', ppl)

sent = '少先队员因该为老人让坐'
sent_seg = segment(sent)
ppl = get_ppl_score(sent_seg)
print('get_ppl_score:', ppl)

print(detect(sent))

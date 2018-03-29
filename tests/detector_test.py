# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from pycorrector import *
from text_util import segment

c = get_same_pinyin('长')
print('get_same_pinyin:', c)

c = get_same_stroke('长')
print('get_same_stroke:', c)

freq = get_frequency('龟龙麟凤')
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

import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

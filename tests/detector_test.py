# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append("../")
from pycorrector.detector import Detector
from pycorrector.tokenizer import segment

d = Detector(enable_rnnlm=True)
sent1 = '少先队员应该为老人让座'
sent_seg = segment(sent1)
ppl = d.ppl_score(sent_seg)
print(sent1, 'ppl_score:', ppl)

sent2 = '少先队员因该为老人让坐'
sent_seg = segment(sent2)
ppl = d.ppl_score(sent_seg)
print(sent2, 'ppl_score:', ppl)

print(sent1, d.detect(sent1))
print(sent2, d.detect(sent2))

freq = d.word_frequency('龟龙麟凤')
print('freq:', freq)

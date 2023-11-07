# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("..")
from pycorrector import Corrector

c = Corrector()
c.check_corrector_initialized()
c.check_detector_initialized()


def get_confusion_set(char):
    confusion_set = c.same_pinyin.get(char, set()).union(c.same_stroke.get(char, set()))
    confusion_set.add(char)
    return confusion_set


def get_score(c1, c2, c3, origin, lambd=1.5):
    ret = c.lm.score(" ".join([c1, c2, c3]))
    if c3 == origin:
        ret *= lambd
    return ret


def trigram_correct(sentence):
    print('sentence:', sentence)
    min_p = 10000
    sentence = list(sentence)
    result = sentence
    for i in range(1, len(sentence) - 1):
        v1 = get_confusion_set(sentence[i - 1])
        v2 = get_confusion_set(sentence[i])
        v3 = get_confusion_set(sentence[i + 1])
        for j in v1:
            for k in v2:
                for l in v3:
                    candidate = sentence[0:i - 1] + list(j) + list(k) + list(l) + sentence[i + 2:]
                    score = c.lm.perplexity(" ".join(candidate))
                    print('candidate, score:', candidate, score)
                    if score < min_p:
                        result = candidate
                        min_p = score
    result = "".join(result)
    print('result:', result)
    return result


if __name__ == '__main__':
    trigram_correct("少先队员因该为老人让坐")

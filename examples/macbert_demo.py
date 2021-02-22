# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from pycorrector import MacBertCorrector

if __name__ == '__main__':

    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
    ]

    m = MacBertCorrector()
    for line in error_sentences:
        correct_sent, err = m.macbert_correct(line)
        print("original sentence:{} => {}, err:{}".format(line, correct_sent, err))

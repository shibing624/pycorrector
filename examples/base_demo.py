# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("..")

import pycorrector

if __name__ == '__main__':
    corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
    print(corrected_sent, detail)

    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
        '这场比赛我甘败下风',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
    ]
    for line in error_sentences:
        correct_sent, err = pycorrector.correct(line)
        print("{} => {} {}".format(line, correct_sent, err))
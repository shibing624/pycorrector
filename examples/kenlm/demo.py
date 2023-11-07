# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../..")
from pycorrector import Corrector

if __name__ == '__main__':
    m = Corrector()
    r = m.correct('少先队员因该为老人让坐')
    print(r)

    error_sentences = [
        '少先队员因该为老人让坐',
        '你找到你最喜欢的工作，我也很高心。',
        '真麻烦你了。希望你们好好的跳无',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
        '这场比赛我甘败下风',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
    ]
    batch_res = m.correct_batch(error_sentences)
    for i in batch_res:
        print(i)
        print()

    # result:
    # {'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让座', 'errors': [('因该', '应该', 4), ('坐', '座', 10)]}
    # {'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}

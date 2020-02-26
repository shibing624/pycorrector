# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")
import pycorrector

pycorrector.set_log_level('INFO')
if __name__ == '__main__':

    error_sentences = [
        '买iPhone差，要多少钱',
        '我想喝小明同学。',
        '哪里卖苹果吧？请大叔给我让坐',
        '交通先行了怎么过去啊？',

    ]
    for line in error_sentences:
        correct_sent = pycorrector.correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

    print('*' * 53)
    pycorrector.set_custom_confusion_dict(path='./my_custom_confusion.txt')
    for line in error_sentences:
        correct_sent = pycorrector.correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

# original sentence:买iPhone差，要多少钱 => correct sentence:('买iPhone差，要多少钱', [])
# original sentence:我想喝小明同学。 => correct sentence:('我想喝小明同学。', [])
# original sentence:哪里卖苹果吧？请大叔给我让坐 => correct sentence:('哪里卖苹果吧？请大叔给我让坐', [])
# original sentence:交通先行了怎么过去啊？ => correct sentence:('交通先行了怎么过去啊？', [])
# *****************************************************
# original sentence:买iPhone差，要多少钱 => correct sentence:('买iphoneX，要多少钱', [['iphone差', 'iphoneX', 1, 8]])
# original sentence:我想喝小明同学。 => correct sentence:('我想喝小茗同学。', [['小明同学', '小茗同学', 3, 7]])
# original sentence:哪里卖苹果吧？请大叔给我让坐 => correct sentence:('哪里卖苹果八？请大叔给我让坐', [['苹果吧', '苹果八', 3, 6]])
# original sentence:交通先行了怎么过去啊？ => correct sentence:('交通限行了怎么过去啊？', [['交通先行', '交通限行', 0, 4]])

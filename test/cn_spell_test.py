# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from pycorrector.cn_spell import *

error_sentence_1 = '机七学习是人工智能领遇最能体现智能的一个分知'
correct_sent, details = correct(error_sentence_1, True)
print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))
for wrong, right, s, e in details:
    print(wrong, right, s, e, )

error_sentence_2 = '杭洲是中国的八大古都之一，因风景锈丽，享有“人间天棠”的美誉！'
correct_sent = correct(error_sentence_2, True)
print("original sentence:{} => correct sentence:{}".format(error_sentence_2, correct_sent))

error_sentence_2 = '我们现今所"使用"的大部分舒学符号，你们用的什么婊点符号'
correct_sent = correct(error_sentence_2, True)
print("original sentence:{} => correct sentence:{}".format(error_sentence_2, correct_sent))

error_sentences = ['按照上级布署安排', '春暖花开之时我们躯车到了海滨渡假村']
for line in error_sentences:
    correct_sent = correct(line, True)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

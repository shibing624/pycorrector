# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import sys

sys.path.append("../")

from pycorrector import correct, get_same_stroke
from pycorrector.utils.math_utils import get_sub_array


def text1():
    error_sentence_1 = '机七学习是人工智能领遇最能体现智能的一个分知'
    correct_sent = correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def text2():
    error_sentence_2 = '杭洲是中国的八大古都之一，因风景锈丽，享有“人间天棠”的美誉！'
    correct_sent = correct(error_sentence_2)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_2, correct_sent))


def text3():
    error_sentence_3 = '我们现今所"使用"的大部分舒学符号，你们用的什么婊点符号'
    correct_sent = correct(error_sentence_3)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_3, correct_sent))


def text4():
    error_sentences = [
        '我喜欢打监球，你呢？足球吗',
        '老师工作非常幸苦,我们要遵敬老师',
        ' 我兴高彩列地去公园游玩',
        '老师的生体不好,可她艰持给我们上课',
        '我们要宝护它们',
        '讲台上放着一只漂亮的刚笔',
        '春暖花开之时我们躯车到了海滨渡假村',
        '按照上级布署安排',
        '冬冬今天戴来了一本好看的童话书',
        '少先队员因该为老人让坐',
        '服装店里的衣服各试各样',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
        ' _ ,',
        '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
        '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',  # 题高 => 提高 专业人氏 => 专业人士
        '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
        '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
        '还有我要看他们的个性，如果跟同时合不来受到压力的话，无法专心地工作。',
    ]
    for line in error_sentences:
        correct_sent = correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))


def homophones():
    nums = [0, 1, 2, 5, 7, 8]
    print(get_sub_array(nums))


def stroke():
    print(get_same_stroke('蓝'))


if __name__ == '__main__':
    text1()
    text2()
    text3()
    text4()
    homophones()
    stroke()

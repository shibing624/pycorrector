# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import re

from zhtools.langconv import Converter
from zhtools.xpinyin import Pinyin


# 去除标点符号
def remove_punctuation(strs):
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())


def traditional2simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def simplified2traditional(sentence):
    '''
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    '''
    sentence = Converter('zh-hant').convert(sentence)
    return sentence


if __name__ == "__main__":
    # traditional simplified
    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = traditional2simplified(traditional_sentence)
    print(simplified_sentence)

    simplified_sentence = '忧郁的台湾乌龟'
    traditional_sentence = simplified2traditional(simplified_sentence)
    print(traditional_sentence)

    # pinyin
    p = Pinyin()
    pinyin = p.get_pinyin('坐骑，你骑哪里了')
    print(pinyin)

    pinyin_tone = p.get_pinyin('坐骑，你骑哪里了', tone=True)
    print(pinyin_tone)

    print(p.get_initials("上"))

    print(''.join(p.py2hz('shang4')))

    print(''.join(p.py2hz('wo')))

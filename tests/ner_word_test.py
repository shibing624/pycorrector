# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
import pycorrector
from pycorrector.utils.tokenizer import segment
error_sentences = [
    '我不要你花钱,这些路曲近通幽', # 曲径通幽
    '这个消息不胫儿走',
    '这个消息不径而走',  # 胫
    '真的是无稽之谈',
    '真的是无集之谈',  # 集
    '肉骨头是索然无味',
    '肉骨头是索染无味',  # 然
    '看书是一心一意，绝不东张夕望，好厉害。',  # 西
    "氨漠索注射液乙基",
    "丙卡特罗片（美普清）乙",
    "瓦贝沙坦技囊（伊泰青）乙省基",
    "复方氨基酸lt（18EAA利泰））甲，限〉基",
    "橘红痰咳液（限）乙省基",
    "兰索拉哇肠溶片乙省基",
    "氯化钾缓釋片甲基",
    "葡萄糖打甲基",
    "小牛曲清去蛋白提取物乙",
    "头抱曲松针（罗氏芬）申基",
    "复方甘草口服溶液限田基",
    '新进人员时，知识当然还不过，可是人有很有精神，面对工作很认真的话，很快就学会、体会。',
]
for line in error_sentences:
    print(line)
    print("segment:", segment(line))
    print(pycorrector.detect(line))
    correct_sent = pycorrector.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

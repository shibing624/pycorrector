# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
from pycorrector.utils.tokenizer import Tokenizer
from pycorrector.utils.tokenizer import segment


def test_segment():
    """测试疾病名纠错"""
    error_sentence_1 = '这个新药奥美砂坦脂片能治疗心绞痛，效果还可以'  # 奥美沙坦酯片
    print(error_sentence_1)
    print(segment(error_sentence_1))
    import jieba
    print(list(jieba.tokenize(error_sentence_1)))
    import jieba.posseg as pseg
    words = pseg.lcut("我爱北京天安门")  # jieba默认模式
    print('old:', words)
    # jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    # words = pseg.cut("我爱北京天安门", use_paddle=True)  # paddle模式
    # for word, flag in words:
    #     print('new:','%s %s' % (word, flag))


def test_tokenizer():
    txts = ["我不要你花钱,这些路曲近通幽",
            "这个消息不胫儿走",
            "这个消息不径而走",
            "这个消息不胫而走",
            "复方甘草口服溶液限田基",
            "张老师经常背课到深夜，我们要体晾老师的心苦。",
            '新进人员时，知识当然还不过，可是人有很有精神，面对工作很认真的话，很快就学会、体会。',
            "小牛曲清去蛋白提取物乙"]
    t = Tokenizer()
    for text in txts:
        print(text)
        print('deault', t.tokenize(text, 'default'))
        print('search', t.tokenize(text, 'search'))
        print('ngram', t.tokenize(text, 'ngram'))

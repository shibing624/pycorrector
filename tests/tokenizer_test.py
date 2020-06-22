# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
from pycorrector.utils.tokenizer import Tokenizer
from pycorrector.utils.tokenizer import segment
from pycorrector.detector import Detector


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
            ",我遇到了问题怎么办",
            ",我遇到了问题",
            "问题",
            "北川景子参演了林诣彬导演的《速度与激情3》",
            "林志玲亮相网友:确定不是波多野结衣？",
            "龟山千广和近藤公园在龟山公园里喝酒赏花",
            "小牛曲清去蛋白提取物乙"]
    t = Tokenizer()
    for text in txts:
        print(text)
        print('deault', t.tokenize(text, 'default'))
        print('search', t.tokenize(text, 'search'))
        print('ngram', t.tokenize(text, 'ngram'))


def test_detector_tokenizer():
    sents = ["我不要你花钱,这些路曲近通幽",
             "这个消息不胫儿走",
             "这个消息不径而走",
             "这个消息不胫而走",
             "复方甘草口服溶液限田基",
             "张老师经常背课到深夜，我们要体晾老师的心苦。",
             '新进人员时，知识当然还不过，可是人有很有精神，面对工作很认真的话，很快就学会、体会。',
             "北川景子参演了林诣彬导演的《速度与激情3》",
             "林志玲亮相网友:确定不是波多野结衣？",
             "龟山千广和近藤公园在龟山公园里喝酒赏花",
             "问题"
             ]
    d = Detector()
    d.check_detector_initialized()
    detector_tokenizer = d.tokenizer
    for text in sents:
        print(text)
        print('deault', detector_tokenizer.tokenize(text, 'default'))
        print('search', detector_tokenizer.tokenize(text, 'search'))

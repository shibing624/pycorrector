# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 用户自定义混淆集，功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
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
        '发行人共同实际控制人萧华、霍荣铨、邓啟棠、张旗康分别',
        '上述承诺内容系本人真实意思表示',
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
# original sentence:发行人共同实际控制人萧华、霍荣铨、邓啟棠、张旗康分别 => correct sentence:('发行人共同实际控制人肖华、霍荣铨、邓啟棠、张启康分别', [['萧华', '肖华', 10, 12], ['张旗康', '张启康', 21, 24]])
# original sentence:上述承诺内容系本人真实意思表示 => correct sentence:('上述承诺内容及本人真实意思表示', [['系', '及', 6, 7]])
# *****************************************************
# original sentence:买iPhone差，要多少钱 => correct sentence:('买iphoneX，要多少钱', [['iphone差', 'iphoneX', 1, 8]])
# original sentence:我想喝小明同学。 => correct sentence:('我想喝小茗同学。', [['小明同学', '小茗同学', 3, 7]])
# original sentence:哪里卖苹果吧？请大叔给我让坐 => correct sentence:('哪里卖苹果八？请大叔给我让坐', [['苹果吧', '苹果八', 3, 6]])
# original sentence:交通先行了怎么过去啊？ => correct sentence:('交通限行了怎么过去啊？', [['交通先行', '交通限行', 0, 4]])
# original sentence:发行人共同实际控制人萧华、霍荣铨、邓啟棠、张旗康分别 => correct sentence:('发行人共同实际控制人萧华、霍荣铨、邓啟棠、张旗康分别', [])
# original sentence:上述承诺内容系本人真实意思表示 => correct sentence:('上述承诺内容系本人真实意思表示', [])

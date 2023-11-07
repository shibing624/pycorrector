# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../..")
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector import ConfusionCorrector

if __name__ == '__main__':
    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
        '他们的吵翻很不错，再说他们做的咖喱鸡也好吃',
        '影像小孩子想的快，学习管理的斑法',
        '餐厅的换经费产适合约会',
        '走路真的麻坊，我也没有喝的东西，在家汪了',
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '我想喝小明同学。',
        '鹅鹅鹅饿',
    ]

    model1 = MacBertCorrector()
    # add confusion corrector for postprocess
    confusion_dict = {"喝小明同学": "喝小茗同学", "老人让坐": "老人让座", "平净": "平静", "分知": "分支"}
    model2 = ConfusionCorrector(custom_confusion_path_or_dict=confusion_dict)
    for line in error_sentences:
        r1 = model1.correct(line)
        correct_sent = r1['target']
        print("query:{} => {} err:{}".format(line, correct_sent, r1['errors']))
        r2 = model2.correct(correct_sent)
        corrected_sent2 = r2['target']
        if corrected_sent2 != correct_sent:
            print("added confusion result: {}".format(r2))

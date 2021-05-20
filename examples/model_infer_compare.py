# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
from pycorrector.bert import bert_corrector
from pycorrector.electra import electra_corrector
from pycorrector.ernie import ernie_corrector
from pycorrector.corrector import Corrector
from pycorrector.macbert import macbert_corrector

error_sentences = [
    '真麻烦你了。希望你们好好的跳无',
    '少先队员因该为老人让坐',
    '少 先  队 员 因 该 为 老人让坐',
    '机七学习是人工智能领遇最能体现智能的一个分知',
    '今天心情很好',
    '汽车新式在这条路上',
    '中国人工只能布局很不错',
    '想不想在来一次比赛',
    '你不觉的高兴吗',
    '权利的游戏第八季',
    '美食美事皆不可辜负，这场盛会你一定期待已久',
    '点击咨询痣疮是什么原因?咨询医师痣疮原因',
    '附睾焱的症状?要引起注意!',
    '外阴尖锐涅疣怎样治疗?-济群解析',
    '洛阳大华雅思 30天突破雅思7分',
    '男人不育少靖子症如何治疗?专业男科,烟台京城医院',
    '疝気医院那好 为老人让坐，疝気专科百科问答',
    '成都医院治扁平苔鲜贵吗_国家2甲医院',
    '少先队员因该为老人让坐',
    '服装店里的衣服各试各样',
    '一只小鱼船浮在平净的河面上',
    '我的家乡是有明的渔米之乡',
    ' _ ,',
    '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
    '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',  # 题高 => 提高 专业人氏 => 专业人士
    '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
    '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
    '今天心情很好',
    '今天新情很好',
]


def main():
    m_rule = Corrector()
    m_bert = bert_corrector.BertCorrector()
    m_electra = electra_corrector.ElectraCorrector()
    m_ernie = ernie_corrector.ErnieCorrector()
    m_macbert = macbert_corrector.MacBertCorrector()
    for line in error_sentences:
        correct_sent, err = m_rule.correct(line)
        print("rule: {} => {}, err:{}".format(line, correct_sent, err))
        correct_sent, err = m_bert.bert_correct(line)
        print("bert: {} => {}, err:{}".format(line, correct_sent, err))
        corrected_sent, err = m_electra.electra_correct(line)
        print("electra: {} => {}, err:{}".format(line, correct_sent, err))
        corrected_sent, err = m_ernie.ernie_correct(line)
        print("ernie: {} => {}, err:{}".format(line, correct_sent, err))
        corrected_sent, err = m_macbert.macbert_correct(line)
        print("macbert: {} => {}, err:{}".format(line, correct_sent, err))
        print()


if __name__ == '__main__':
    main()

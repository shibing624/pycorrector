# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from pycorrector import corrector

error_sentences = [
    '美食美事皆不可辜负，这场盛会你一定期待已久',
    '点击咨询痣疮是什么原因?咨询医师痣疮原因',
    '附睾焱的症状?要引起注意!',
    '外阴尖锐涅疣怎样治疗?-济群解析',
    '洛阳大华雅思 30天突破雅思7分',
    '男人不育少靖子症如何治疗?专业男科,烟台京城医院',
    '疝気医院那好 疝気专科百科问答',
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
]
for line in error_sentences:
    correct_sent = corrector.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
# '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',  # 题高 => 提高 专业人氏 => 专业人士
# '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
# 天地无垠	天地无限
# 方大碳素等等	方大炭素等等
# 耐得住欲妄	耐得住欲望
# 交通先行	交通限行
# 苹果吧   苹果八

import pycorrector

error_sentences = [
    '买iPhone差多少钱',
    '我想喝小明同学。',
    '哪里卖苹果吧？请大叔给我让坐',
    '我对于宠物出租得事非常认同',
    '天地无垠大，我们的舞台无线大',
    '交通先行了怎么过去啊？',

]
for line in error_sentences:
    # idx_errors = pycorrector.detect(line)
    # print(idx_errors)

    correct_sent = pycorrector.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

print('*' * 53)
pycorrector.set_custom_confusion_dict(path='./my_confusion.txt')
for line in error_sentences:
    # idx_errors = pycorrector.detect(line)
    # print(idx_errors)
    correct_sent = pycorrector.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

# original sentence:买iPhone差多少钱 => correct sentence:('买iPhone此多少钱', [['差', '此', 7, 8]])
# original sentence:我想喝小明同学。 => correct sentence:('我像喝晓明同学。', [['我想', '我像', 0, 2], ['喝小明', '喝晓明', 2, 5]])
# original sentence:哪里卖苹果吧？请大叔给我让坐 => correct sentence:('哪里卖苹果吧？请大叔给我让座', [['坐', '座', 13, 14]])
# original sentence:我对于宠物出租得事非常认同 => correct sentence:('我对于宠物出租得是非常认同', [['得事', '得是', 7, 9]])
# original sentence:天地无垠大，我们的舞台无线大 => correct sentence:('天地无限贷，我们的舞台无限大', [['天地无垠', '天地无限', 0, 4], ['大', '贷', 4, 5], ['无线大', '无限大', 11, 14]])
# original sentence:交通先行了怎么过去啊？ => correct sentence:('交通先行了怎么过去啊？', [])
# *****************************************************
# original sentence:买iPhone差多少钱 => correct sentence:('买iphoneX多少钱', [['iphone差', 'iphoneX', 1, 8]])
# original sentence:我想喝小明同学。 => correct sentence:('我想喝小茗同学。', [['小明同学', '小茗同学', 3, 7]])
# original sentence:哪里卖苹果吧？请大叔给我让坐 => correct sentence:('哪里卖苹果八？请大叔给我让座', [['苹果吧', '苹果八', 3, 6], ['坐', '座', 13, 14]])
# original sentence:我对于宠物出租得事非常认同 => correct sentence:('我对于宠物出租的事非常认同', [['得事', '的事', 7, 9]])
# original sentence:天地无垠大，我们的舞台无线大 => correct sentence:('天地无限贷，我们的舞台无限大', [['天地无垠', '天地无限', 0, 4], ['大', '贷', 4, 5], ['无线大', '无限大', 11, 14]])
# original sentence:交通先行了怎么过去啊？ => correct sentence:('交通限行了怎么过去啊？', [['交通先行', '交通限行', 0, 4]])

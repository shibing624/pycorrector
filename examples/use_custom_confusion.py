# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 用户自定义混淆集，功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
"""

import sys

sys.path.append("../")
import pycorrector

# pycorrector.set_log_level('INFO')
if __name__ == '__main__':

    error_sentences = [
        '买iphonex，要多少钱',  # 漏召回
        '我想喝小明同学。',  # 漏召回
        '哪里卖苹果吧？请大叔给我让坐',  # 漏召回
        '交通先行了怎么过去啊？',  # 漏召回
        '共同实际控制人萧华、霍荣铨、张旗康',  # 误杀
        '上述承诺内容系本人真实意思表示',  # 正常
        '大家一哄而伞怎么回事',  # 成语
    ]
    for line in error_sentences:
        print(pycorrector.correct(line))

    print('*' * 42)
    pycorrector.set_custom_confusion_dict(path='./my_custom_confusion.txt')
    for line in error_sentences:
        print(pycorrector.correct(line))

# ('买iphonex，要多少钱', [])
# ('我想喝小明同学。', [])
# ('哪里卖苹果吧？请大叔给我让坐', [])
# ('交通先行了怎么过去啊？', [])
# ('共同实际控制人萧华、霍荣铨、张启康', [['张旗康', '张启康', 14, 17]])
# ('上述承诺内容系本人真实意思表示', [])
# *****************************************************
# ('买iphoneX，要多少钱', [['iphonex', 'iphoneX', 1, 8]])
# ('我想喝小茗同学。', [['小明同学', '小茗同学', 3, 7]])
# ('哪里卖苹果八？请大叔给我让坐', [['苹果吧', '苹果八', 3, 6]])
# ('交通限行了怎么过去啊？', [['交通先行', '交通限行', 0, 4]])
# ('共同实际控制人萧华、霍荣铨、张旗康', [])
# ('上述承诺内容系本人真实意思表示', [])

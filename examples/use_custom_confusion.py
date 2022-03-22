# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 用户自定义混淆集，功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
"""

import sys

sys.path.append("..")
from pycorrector import Corrector, set_log_level

set_log_level('INFO')
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
    m = Corrector()
    for i in error_sentences:
        print(i, ' -> ', m.correct(i))

    print('*' * 42)
    m = Corrector(custom_confusion_path='./my_custom_confusion.txt')
    for i in error_sentences:
        print(i, ' -> ', m.correct(i))

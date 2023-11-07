# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自定义成语纠错
"""

import sys

sys.path.append("../..")
from pycorrector import Corrector

if __name__ == '__main__':
    error_sentences = [
        '这块名表带带相传',
        '这块名表代代相传',
        '他贰话不说把牛奶喝完了',
        '这场比赛我甘败下风',
        '这场比赛我甘拜下封',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
        '有了这一番旁证博引',
        '有了这一番旁针博引',
        '这群鸟儿迁洗到远方去了',
        '这群鸟儿千禧到远方去了',
        '大家一哄而伞怎么回事',  # 成语
        '我想去长江达桥走一走',
        '美国前总统特琅普给普京点了一个赞，特朗普称普金做了一个果断的决定',
        '今天在拼哆哆上买了点苹果',
        '消炎可以吃点阿木西林药品',  # 阿莫西林
    ]
    m = Corrector(proper_name_path='')
    for i in error_sentences:
        print(i, ' -> ', m.correct(i))

    print('*' * 42)
    m = Corrector(proper_name_path='./my_custom_proper.txt')
    for i in error_sentences:
        print(i, ' -> ', m.correct(i))

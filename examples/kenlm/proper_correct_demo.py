# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
sys.path.append("../..")
from pycorrector import ProperCorrector

if __name__ == '__main__':
    """成语纠错"""
    m = ProperCorrector()
    # 报应接中迩（（而））来。
    # 人群穿（（川））流不息。
    # 这个消息不径（（胫））而走。
    # 这家伙还蛮格（（恪））尽职守的。
    # 眼前的场景美仑（（轮））美幻简直超出了人类的想象。
    # 看着这两个人谈笑风声（（生））我心理（（里））不由有些忌妒。
    # 有了这一番旁证（（征））博引。
    x = [
        '报应接中迩来',
        '这块名表带带相传',
        '今天在拼哆哆上买了点苹果',
        '他贰话不说把牛奶喝完了',
        '这场比赛我甘败下风',
        '这场比赛我甘拜下封',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
    ]

    for i in x:
        print(i, ' -> ', m.correct(i))

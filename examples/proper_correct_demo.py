# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from pycorrector.proper_corrector import ProperCorrector
from pycorrector import config

if __name__ == '__main__':
    """成语纠错"""
    m = ProperCorrector(proper_name_path=config.proper_name_path)
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
        '这块名表代代相传',
        '他贰话不说把牛奶喝完了',
        '这场比赛我甘败下风',
        '这场比赛我甘拜下封',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
        '人群穿流不息',
        '这个消息不径而走',
        '这个消息不胫儿走',
        '眼前的场景美仑美幻简直超出了人类的想象',
        '看着这两个人谈笑风声我心理不由有些忌妒',
        '有了这一番旁证博引',
        '有了这一番旁针博引',
        '这群鸟儿迁洗到远方去了',
        '这群鸟儿千禧到远方去了',
        '美国前总统特琅普给普京点了一个赞，特朗普称普金做了一个果断的决定',
    ]

    for i in x:
        print(i, ' -> ', m.proper_correct(i))

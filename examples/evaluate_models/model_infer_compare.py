# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../..")
from pycorrector import Corrector, MacBertCorrector


def main():
    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '我的家乡是有明的渔米之乡',
        ' _ ,',
        '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
        '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',
        # 题高 => 提高 专业人氏 => 专业人士
        '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
        '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
        '今天心情很好',
        '今天新情很好',
    ]
    m_kenlm = Corrector()
    m_macbert = MacBertCorrector()
    for line in error_sentences:
        r = m_kenlm.correct(line)
        print("kenlm: {}".format(r))
        r = m_macbert.correct(line)
        print("macbert: {}".format(r))
        print()


if __name__ == '__main__':
    main()

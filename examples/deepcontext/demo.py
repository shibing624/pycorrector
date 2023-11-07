# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')

from pycorrector import DeepContextCorrector


def main():
    m = DeepContextCorrector()
    error_sentences = [
        '今天新情很好',
        '你找到你最喜欢的工作，我也很高心。',
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    batch_res = m.correct_batch(error_sentences)
    for i in batch_res:
        print(i)
        print()

    # result:
    # {'source': '今天新情很好', 'target': '今天心情很好', 'errors': [('新', '心', 2)]}
    # {'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}


if __name__ == "__main__":
    main()

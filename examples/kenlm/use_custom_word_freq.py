# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 用户自定义混淆集，功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
"""

import sys

sys.path.append("../..")
import pycorrector

if __name__ == '__main__':
    error_sentences = [
        '根据联合国公布数据显示，全球产龄妇女从１９５０年６．３亿人增至２０１０年17亿人，由统计数据显示，至２０５０年全球产龄妇女将达20亿人',
        '晓红的藉贯是北京。',
        '双十一下单到现在还没发货的',
        '汽车行试在这条路上'
    ]
    for line in error_sentences:
        print(pycorrector.correct(line))

    print('*' * 42)
    pycorrector.set_custom_word_freq(path='./custom_word_freq.txt')
    for line in error_sentences:
        print(pycorrector.correct(line))

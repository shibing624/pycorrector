# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")

import pycorrector

if __name__ == '__main__':
    sent = '少先队员因该为老人让坐'
    corrected_sent, detail = pycorrector.correct(sent)
    print(corrected_sent, detail)

    print("*" * 42)
    pycorrector.enable_char_error(enable=False)
    correct_sent, detail = pycorrector.correct(sent)
    print("{} => {} {}".format(sent, correct_sent, detail))

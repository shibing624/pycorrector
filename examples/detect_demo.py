# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")

import pycorrector

pycorrector.set_log_level('INFO')
if __name__ == '__main__':
    idx_errors = pycorrector.detect('少先队员因该为老人让坐')
    print(idx_errors)

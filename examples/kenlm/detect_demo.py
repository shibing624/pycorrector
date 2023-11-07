# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../..")

from pycorrector import Corrector

if __name__ == '__main__':
    m = Corrector()
    idx_errors = m.detect('少先队员因该为老人让坐')
    print(idx_errors)

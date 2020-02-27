# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("../")

import tracemalloc

tracemalloc.start(10)
time1 = tracemalloc.take_snapshot()

import pycorrector

idx_errors = pycorrector.detect('少先队员因该为老人让坐')
print(idx_errors)

time2 = tracemalloc.take_snapshot()
stats = time2.compare_to(time1, 'lineno')
print('*' * 32)
for stat in stats[:3]:
    print(stat)

stats = time2.compare_to(time1, 'traceback')
print('*' * 32)
for stat in stats[:3]:
    print(stat.traceback.format())

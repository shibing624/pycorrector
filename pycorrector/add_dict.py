# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import pdb
sys.path.append('../')
import difflib
from collections import defaultdict

file = codecs.open('../tests/correction.txt', 'rb', encoding = 'utf-8').readlines()
dic  = codecs.open('data/same_pinyin.txt', 'w+', encoding = 'utf-8').readlines()

dict_ = defaultdict(set)


# print(file[-5:])
for line in file:
    print(line)
    wrong, right = line.split()

    for i in range(len(wrong)):
        if wrong[i] != right[i]:
            dict_[wrong[i]].add(right[i])
print(dict_)
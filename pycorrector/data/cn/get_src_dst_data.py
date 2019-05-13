# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys
from nltk.tokenize import  word_tokenize

type = 'src'
for line in sys.stdin:
    line = line.strip()
    if type == 'src':
        if line[:4] == "src:":
            print(" ".join(word_tokenize(line[5:])))
    else:
        print(" ".join(list(line[5:])))
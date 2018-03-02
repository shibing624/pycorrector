# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import kenlm
import jieba
import pickle
import math
import wubi
import numpy as np
import pypinyin
from pypinyin import pinyin
from collections import Counter
import config
import re



def main():
    line = '我们现在使用的数学福号'
    print('input sentence is:'%line)
    # corrected_sent,correct_ranges = correct(line)

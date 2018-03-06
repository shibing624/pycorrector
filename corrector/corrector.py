# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: main

import os
import kenlm
import jieba
import pickle
import math
import wubi
import numpy as np
from pypinyin import pinyin
from collections import Counter
import config
import re

print('Loading models...')
jieba.initialize()

bimodel_path = ""
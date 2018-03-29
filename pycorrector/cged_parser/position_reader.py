# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import os
import sys
from pycorrector.io_util import default_logger

from xml.dom import minidom

error_dict = {
    'R':1, # 重复词
    'M':2, # 缺词
    'S':3, # 用错词
    'W':4, # 词序错误
}

def train_reader(data_path):
    pass

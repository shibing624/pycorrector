# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pycorrector import detector

# fixed
idx_errors = detector.detect('vd')
print(idx_errors)

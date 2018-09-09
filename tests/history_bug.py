# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import pycorrector

corrected_sent, detail = pycorrector.correct('cv')
print(corrected_sent, detail)
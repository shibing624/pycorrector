# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import paddle
paddle_version = [int(i) for i in paddle.__version__.split('.')]
if paddle_version[1] < 7:
    raise RuntimeError('paddle-ernie requires paddle 1.7+, got %s' %
                       paddle.__version__)

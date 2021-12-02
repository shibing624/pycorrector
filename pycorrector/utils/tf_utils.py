# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os

import tensorflow as tf


def get_ckpt_path(model_path):
    ckpt = tf.train.get_checkpoint_state(model_path)
    ckpt_path = ""
    if ckpt:
        ckpt_file = ckpt.model_checkpoint_path.split('/')[-1]
        ckpt_path = os.path.join(model_path, ckpt_file)
    return ckpt_path

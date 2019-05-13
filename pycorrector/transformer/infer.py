# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')

from pycorrector.transformer import config
from pycorrector.transformer.model import translate, source_inputter, target_inputter

import opennmt as onmt

if __name__ == "__main__":
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": config.vocab_path,
        "target_vocabulary": config.vocab_path
    })
    translate(config.model_dir,
              inputter,
              config.src_test_path,
              batch_size=config.batch_size,
              beam_size=config.beam_size)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

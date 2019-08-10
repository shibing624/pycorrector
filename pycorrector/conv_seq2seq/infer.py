# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import sys

from fairseq import options

sys.path.append('../..')

from pycorrector.conv_seq2seq import config
from pycorrector.conv_seq2seq import interactive


def infer(model_path, vocab_dir, arch, test_data, max_len):
    parser = options.get_generation_parser(interactive=True)
    parser.set_defaults(arch=arch,
                        input=test_data,
                        max_tokens=max_len,
                        path=model_path)
    args = options.parse_args_and_arch(parser, input_args=[vocab_dir])
    return interactive.main(args)


def infer_interactive(model_path, vocab_dir, arch, max_len):
    return infer(model_path, vocab_dir, arch, '-', max_len)


if __name__ == '__main__':
    # 通过文本预测
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
    outputs = infer(model_path=config.best_model_path,
                    vocab_dir=config.data_bin_dir,
                    arch=config.arch,
                    test_data=[' '.join(list(i)) for i in inputs],
                    max_len=config.max_len)
    print("output:", outputs)

    # 通过文件预测
    outputs = infer(model_path=config.best_model_path,
                    vocab_dir=config.data_bin_dir,
                    arch=config.arch,
                    test_data=config.val_src_path,
                    max_len=config.max_len)
    print("output:", outputs)



# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from pycorrector.seq2seq.seq2seq_corrector import Seq2SeqCorrector

if __name__ == "__main__":
    m = Seq2SeqCorrector()
    error_sentences = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    res = m.seq2seq_correct(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} , err:{}".format(sent, r[0], r[1]))

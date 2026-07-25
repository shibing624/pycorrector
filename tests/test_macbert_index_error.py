# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

import pytest

sys.path.append('..')
from pycorrector import MacBertCorrector


@pytest.mark.heavy(repo='shibing624/macbert4csc-base-chinese')
def test_macbert_index():
    m = MacBertCorrector()
    sents = [
        '我们禅精竭虑学习',
        '禅精竭虑学习',
    ]
    res = []
    for i in sents:
        r = m.correct(i)
        print(i, r)
        res.append(r)

    assert res[0]['target'] == '我们禅精竭虑学习'
    assert res[1]['target'] == '禅精竭虑学习'

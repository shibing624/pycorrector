# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from pycorrector.macbert.macbert_corrector import MacBertCorrector


def test_long_text_for_macbert():
    sents = ['6、在陈刚担任公司董事、监事、高级管理人员期间，每年转让本公司持有的公司股份数量不超过直接或间接持有公司股份总数的25%，' \
             '所持股份总数不超过1,000股的除外；在陈刚离职六个月内，不转让本公司所持有的公司股份；陈刚在担任公司董事、监事、高级管理' \
             '人员任期届满前离职的，在其就任时确定的任期内和任期届满后6个月内，本公司每年转让持有的公司股份数量不超过直接或间接持有' \
             '公司股份总数的25%，离职陈刚后半年内，本公司不转让所持本公司股份。',
             ]
    m = MacBertCorrector()
    for line in sents:
        correct_sent, err = m.macbert_correct(line)
        print("sentence:{} => {} err:{}".format(line, correct_sent, err))

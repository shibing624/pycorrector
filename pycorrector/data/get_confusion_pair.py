# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from pycorrector.utils.eval import get_bcmi_corpus

data_path = 'errors.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # 甘败(拜)下风 => 甘败下风	甘拜下风
        error_sentence, right_sentence, right_detail = get_bcmi_corpus(line, left_symbol='(', right_symbol=')')
        if not error_sentence:
            continue
        # print(right_detail)
        print(error_sentence + '\t' + right_sentence)

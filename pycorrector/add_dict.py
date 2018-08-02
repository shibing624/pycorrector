# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import pdb
sys.path.append('../')

from pycorrector.corrector import load_same_pinyin

new_char = codecs.open('../sighan8_test/sighan8_result/correction.txt', \
                       'rb', encoding = 'utf-8').readlines()
old_char = codecs.open('data/same_stroke.txt',   \
                       'rb', encoding = 'utf-8').readlines()

set_stroke  = [set(i[:-1].split(',')) for i in old_char]
dict_pinyin = load_same_pinyin('data/same_pinyin.txt')

new_file = codecs.open('data/same_stroke.txt', 'w+', encoding = 'utf-8')

for line in new_char:
    wrong, right = line.split()

    for i in range(len(wrong)):
        if wrong[i] != right[i]:
            if wrong[i] not in dict_pinyin or right[i] not in dict_pinyin[wrong[i]]:
                flag = 0
                for sets in set_stroke:
                    if wrong[i] in sets:
                        sets.add(right[i])
                        flag = 1

                    elif right[i] in sets:
                        sets.add(wrong[i])
                        flag = 1

                if flag == 0:
                    set_stroke.append(set([wrong[i], right[i]]))


for sets in set_stroke:
    new_file.write(','.join(list(sets)) + '\n')

new_file.close()


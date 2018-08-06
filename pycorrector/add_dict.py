# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import pdb
sys.path.append('../')

from pycorrector.corrector import load_same_pinyin

def main():
    # # laoding files
    result   = codecs.open('../sighan8_test/sighan8_result/compare_result.tmp', \
                           'rb', encoding = 'utf-8').readlines()
    old_char = codecs.open('data/same_stroke.txt.bak',   \
                           'rb', encoding = 'utf-8').readlines()
    set_stroke  = [set(i[:-1].split(',')) for i in old_char]
    dict_pinyin = load_same_pinyin('data/same_pinyin.txt')

    new_file = codecs.open('data/same_stroke.txt', 'w+', encoding = 'utf-8')



    for line in result:
        if line[:11] == 'true_change' and len(line) > 16:
            change_list = [change.split('-->') for change in \
                           line[14:].strip().replace(' ', '').split(',') if change]
            for [wrong, right] in change_list:

                if wrong not in dict_pinyin or right not in dict_pinyin[wrong]:
                    flag = 0
                    # for sets in set_stroke:
                    #     if wrong in sets:
                    #         sets.add(right)
                    #         flag = 1

                    #     elif right in sets:
                    #         sets.add(wrong)
                    #         flag = 1

                    for sets in set_stroke:
                        if wrong in sets and right in sets:
                            flag = 1


                    if flag == 0:
                        set_stroke.append(set([wrong, right]))


    for sets in set_stroke:
        new_file.write(','.join(list(sets)) + '\n')

    new_file.close()

if __name__ == '__main__':
    main()

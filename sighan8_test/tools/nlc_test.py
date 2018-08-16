# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import pdb

# cor_file = codecs.open('/Users/kun/xiaoI/opentool/Neural_Language_Correction/nlpcc_data/data/test_sighan8/test_sent.txt', 'rb' , encoding = 'utf-8').readlines()
cor_file = codecs.open('test_sent.txt', 'rb' , encoding = 'utf-8').readlines()

err_file = codecs.open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestInput.txt', 'rb', encoding = 'utf-8').readlines()
result_file = codecs.open('sighan8_result/corrected_nlc_result.txt',   'w+', encoding = 'utf-8')



sys.stderr.write('REFORMING RESULTS : start formatting correction result\n')

for i, line in enumerate(err_file):
    pid_e, err_sent = line.split('\t')
    cor_sent = cor_file[i]

    result = pid_e[5:-1]


    for idx in range(len(cor_sent)):
        if idx < len(err_sent) and cor_sent[idx] != err_sent[idx]:
            # # extracting corrected char
            result += ', ' + str(idx + 1) + ', ' + cor_sent[idx]
        # elif idx >= len(err_sent):
        	


    # # if with no change, just output location 0
    if result == pid_e[1:-1]:
        result += ', 0'
    result += '\n'

    result_file.write(result)

result_file.close()

sys.stderr.write('REFORMING RESULTS : finishing formatting correction result\n')

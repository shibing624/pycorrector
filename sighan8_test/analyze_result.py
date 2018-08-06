# -*- coding: utf-8 -*-
#!/usr/bin/env python
#

import re
import pdb
import argparse
from codecs import open
from tqdm import tqdm
import sys
sys.path.append("../")

def main():
    cor_sent = open('sighan8_result/corrected_sentence.txt', 'rb', encoding = 'utf-8').readlines()
    err_sent = open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestInput.txt', 'rb', encoding = 'utf-8').readlines()
    pred_change = open('sighan8_result/corrected_result.txt', 'rb', encoding = 'utf-8').readlines()
    true_change = open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestTruth.txt', 'rb', encoding = 'utf-8').readlines()

    comp_file = open('sighan8_result/compare_result.tmp', 'w+', encoding = 'utf-8')

    # print(len(cor_sent),len(err_sent),len(pred_change),len(true_change))

    if not len(cor_sent) == len(err_sent) == len(pred_change) == len(true_change):
        sys.stderr.write("ERROR: four files' length not match")
        quit()

    for i in range(len(cor_sent)):
        pred_change_line = pred_change[i].strip().split(', ')[1:]

        pred_detail = ''

        if len(pred_change_line) > 1:
            # print(len(pred_change_line))
            for idx in range(int(len(pred_change_line) / 2)):
                pred_detail += err_sent[i].split('\t')[1][int(pred_change_line[idx * 2]) - 1] + \
                                    ' --> '  + pred_change_line[idx * 2 + 1] + ', '

        true_change_line = true_change[i].strip().split(', ')[1:]
        true_detail = ''
        if len(true_change_line) > 1:
            # print(len(true_change_line))
            for idx in range(int(len(true_change_line) / 2)):
                true_detail += err_sent[i].split('\t')[1][int(true_change_line[idx * 2]) - 1] + \
                                    ' --> '  + true_change_line[idx * 2 + 1] + ', '

        # sys.stderr.write('input_sentence :  ' + err_sent[i].split('\t')[1])
        # sys.stderr.write('output_sentence : ' + cor_sent[i].split('\t')[1])
        # sys.stderr.write('pred_change : ' + pred_detail + '\n')
        # sys.stderr.write('true_change : ' + true_detail + '\n')

        comp_file.write('input_sentence  : ' + err_sent[i].split('\t')[1])
        comp_file.write('output_sentence : ' + cor_sent[i].split('\t')[1])
        comp_file.write('pred_change : ' + pred_detail + '\n')
        comp_file.write('true_change : ' + true_detail + '\n')
    comp_file.close()



if __name__ == '__main__':
    main()






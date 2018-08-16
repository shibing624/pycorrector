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
    file_sent   = open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestInput_mod.txt', 
                       'rb', encoding = 'utf-8').readlines()

    file_change = open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestTruth_mod.txt',
                       'rb', encoding = 'utf-8').readlines()

    for idx, line in enumerate(file_change):
        changes = line[: -1].split(', ')
        num_change = int((len(changes) - 1) / 2 )

        if num_change > 0 and line[: 9] == file_sent[idx][5: 14]:

            for i in range(num_change):

                file_sent[idx] = file_sent[idx][: 15 + int(changes[1:][i * 2])] + \
                                 changes[1:][i * 2 + 1] + \
                                 file_sent[idx][15 + int(changes[1:][i * 2]) + 1:]

        elif line[: 9] != file_sent[idx][5: 14]:
            sys.stderr.write('*************  miss ordered   ************')

    file_output = open('sighan8csc_release1.0/Test/SIGHAN15_CSC_TestTruth_Sentence.txt', 
                       'w+', encoding = 'utf-8')

    file_output.write(''.join(file_sent))
    file_output.close()






if __name__ == '__main__':
	main()
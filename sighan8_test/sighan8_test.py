# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import argparse
import sys
sys.path.append("../")

from codecs import open
from tqdm import tqdm
from pycorrector.corrector import correct

def parse():
    parser = argparse.ArgumentParser(
             description = 'this file is to use pycorrector to test '
                           'sighan15 test file, and transfer the result'
                           'to the format that sighan15 eval tool required')
    parser.add_argument('-e', '--error_sentence', #required = True, 
                        default = 'sighan8csc_release1.0/Test/SIGHAN15_CSC_TestInput.txt',
                        help = 'error sentenced to be corrected'
                               '(format should be the same as sighan15_test_input)')
    parser.add_argument('-c', '--corrected_sentence', #required = True,
                        default = 'sighan8_result/corrected_sentence.txt',
                        help = 'file to store corrected sentence(not required)')
    parser.add_argument('-r', '--result', #required = True,
                        default = 'sighan8_result/corrected_result.txt',
                        help = 'file to store correction detail, in the format'
                               'sighan15 eval tool required')
    parser.add_argument('-v', '--correct_verbose',
                        default = False,
                        help = 'show the detail of correction or not')
    parser.add_argument('--param_ec', type = float,
                        default = 1.4,
                        help = 'parameter for adjust the weight of edition cost')
    parser.add_argument('--param_gd', type = float,
                        default = 2,
                        help = 'parameter for adjust the weight of global decision')
    return parser.parse_args()

def eval_sighan(input_path, output_path, param_ec, param_gd, verbose=False):
    '''
    Input:
        input_path:  file of original sentences      form: (pid)\terror_sentence
        output_path: path of predicted sentences     form: (pid)\tcorrected_sentence
        verbose:     print the error and corrected sentences during running or not
    '''
    
    sys.stderr.write('sighan15_test            : start correcting sentences......\n')
    sys.stderr.write('error_sentences_path     : ' + input_path + '\n')
    sys.stderr.write('corrected_sentences_path : ' + output_path + '\n')
    sighan_data = open(input_path, 'rb', encoding = 'utf-8')
    corr_file   = open(output_path, 'w+', encoding = 'utf-8')

    if verbose:
        for line in sighan_data.readlines():
            pid, sentence = line.split('\t')
            pred_sent, pred_detail = correct(sentence.strip(), param_ec, param_gd)

            sys.stderr.write('input sentence : ' + sentence)
            sys.stderr.write('pred sentence  : ' + pred_sent)
            sys.stderr.write('predict change : ' + ', '.join([i[0][0] + '-->' + i[0][1] \
                                       for i in pred_detail if i]) + '\n')

            corr_file.write(pid + '\t' + pred_sent + '\n')
    else:
        for line in tqdm(sighan_data.readlines()):
            pid, sentence = line.split('\t')
            pred_sent, pred_detail = correct(sentence.strip(), param_ec, param_gd)

            corr_file.write(pid + '\t' + pred_sent + '\n')

    corr_file.close()
    sighan_data.close()

    sys.stderr.write('sighan15_test            : finishing correcting sentences\n')


def format_result(err_sent_path, cor_sent_path, result_path):
    '''
    Input: 
        err_sent_path: file of original sentences      form: (pid)\terror_sentence
        cor_sent_path: file of predicted sentences     form: (pid)\tcorrected_sentence
        result_path:   path of result in the format that sighan15 required
                                                       form: pid((, #, char)+|, 0)\n
    '''

    sys.stderr.write('REFORMING RESULTS : start formatting correction result\n')

    err_file    = open(err_sent_path, 'rb', encoding = 'utf-8').readlines()
    cor_file    = open(cor_sent_path, 'rb', encoding = 'utf-8').readlines()
    result_file = open(result_path,   'w+', encoding = 'utf-8')

    for i, line in enumerate(err_file):
        pid_e, err_sent = line.split('\t')
        pid_c, cor_sent = cor_file[i].split('\t')

        result = pid_e[5:-1]

        # # matching pid
        if pid_e == pid_c:
            if len(err_sent) != len(cor_sent):
                result += ',0'
            else:
                for idx in range(len(cor_sent)):

                    if cor_sent[idx] != err_sent[idx]:
                    	# # extracting corrected char
                        result += ', ' + str(idx + 1) + ', ' + cor_sent[idx]
        # # if pid not matched, go over all the cor_sent to find 
        # # corresponding sentence.
        else:
            for sent in cor_file:
                if pid_e == sent.split('\t')[0]:
                    for idx in range(len(cor_sent)):
                        if cor_sent[idx] != err_sent[idx]:
                            result += ', ' + str(idx + 1) + ', ' + cor_sent[idx]

        # # if with no change, just output location 0
        if result == pid_e[5:-1]:
            result += ', 0'
        result += '\n'

        result_file.write(result)

    result_file.close()

    sys.stderr.write('REFORMING RESULTS : finishing formatting correction result\n')

if __name__ == "__main__":
    args = parse()

    eval_sighan  (args.error_sentence, 
                  args.corrected_sentence,
                  args.param_ec,
                  args.param_gd,
                  args.correct_verbose)

    format_result(args.error_sentence, 
                  args.corrected_sentence, 
                  args.result)









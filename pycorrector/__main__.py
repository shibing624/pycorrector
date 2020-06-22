# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import argparse

import pycorrector


def main(**kwargs):
    """
    Cmd script of correct. Input text file, output corrected text file.
    :param kwargs: input, a text file object that will be read from. Should contain utf-8 sentence per line
    :param output: a text file object where parsed output will be written. Parsed output will be similar to CSV data
    :type input: text file object in read mode
    :type output: text file object in write mode
    :return:
    """
    no_char = kwargs['no_char'] if 'no_char' in kwargs else False
    if no_char:
        pycorrector.enable_char_error(enable=False)
        print('disable char error detect.')

    detail = kwargs['detail'] if 'detail' in kwargs else False
    count = 0
    with open(kwargs['input'], 'r', encoding='utf-8') as fr, open(kwargs['output'], 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            corrected_sent, info = pycorrector.correct(line)
            count += 1
            r = corrected_sent
            if detail:
                r = corrected_sent + '\t' + str(info)
            fw.write(line + '\t' + r + '\n')
        print('{} lines in output'.format(count))


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=str,
                        help='the input file path, file encode need utf-8.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='the output file path.')
    parser.add_argument('-n', '--no_char', action="store_true", help='disable char detect mode.')
    parser.add_argument('-d', '--detail', action="store_true", help='print detail info')
    args = parser.parse_args()
    print(args)
    main(**vars(args))


if __name__ == '__main__':
    run()

# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import argparse
import pdb
sys.path.append("../")
from pycorrector.utils.text_utils import is_chinese
from pycorrector.utils.text_utils import traditional2simplified
from pycorrector.utils.io_utils import get_logger

# if len(sys.argv) == 1:
#     sys.argv = ['tra2sim', 'data/test', 'data/test_sim']

def parse():
    parser = argparse.ArgumentParser(description = 'this is for tokenize file with one sentence in each line')
    
    parser.add_argument('-i','--input_file', required = True,
                        help = 'file to be processed')
    parser.add_argument('-o','--output_file', required = True,
                        help = 'file to store processed setence')
    parser.add_argument('-e','--effect', default = False,
                        help = 'judge the sentence is effective or not (including unknown symbol or char)')
    return parser.parse_args()


def main():
    args = parse()
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(pwd_path, "/" + args.input_file)

    if not os.path.exists(file_path):
        default_logger = get_logger(__file__)
        default_logger.debug("file not exists:", file_path)

    file_in = codecs.open(args.input_file, 'rb', encoding = 'utf-8').readlines()
    file_ou = codecs.open(args.output_file, 'w', encoding = 'utf-8')

    if args.effect:
        PUNCTUATION_LIST = "。，,、？：；{}[]【】“‘’”《》/！%……（）<>@#$~^￥%&*\"\'=+-"
        for line in file_in:
            line = line.strip()
            if False not in [(char in PUNCTUATION_LIST or is_chinese(char)) for char in line]:
                line = traditional2simplified(line)
                file_ou.write(line + '\n')
        file_ou.close()
    else:
        for line in file_in:
            line = line.strip()
            line = traditional2simplified(line)
            file_ou.write(line + '\n')
        file_ou.close()


if __name__ == '__main__':
    main()


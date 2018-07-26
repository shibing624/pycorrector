# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
import sys
import os
import codecs
import pdb
sys.path.append("../")
from pycorrector.utils.text_utils import traditional2simplified
from pycorrector.utils.io_utils import get_logger

if len(sys.argv) == 1:
    sys.argv = ['tra2sim', 'data/test', 'data/test_sim']



def main():
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(pwd_path, "/" + sys.argv[1])

    if not os.path.exists(file_path):
        default_logger = get_logger(__file__)
        default_logger.debug("file not exists:", file_path)

    file_in = codecs.open(sys.argv[1], 'rb', encoding = 'utf-8').readlines()
    file_ou = codecs.open(sys.argv[2], 'w', encoding = 'utf-8')

    for line in file_in:
        line = traditional2simplified(line.strip())
        file_ou.write(line + '\n')
    file_ou.close()




if __name__ == '__main__':
    main()


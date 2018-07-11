# -*- coding: utf-8 -*-
#！/usr/bin/env python
#
import os
import sys
sys.path.append("../")
import re
from codecs import open
from pycorrector.corrector import correct
from pycorrector.utils.io_utils import load_pkl

from tqdm import tqdm
import pdb



pwd_path  = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(pwd_path, '../pycorrector/data/test/source.txt')
pred_path = os.path.join(pwd_path, '../pycorrector/data/test/prediction.txt')
	
input_file  = open(data_path, 'rb', encoding = 'utf-8').readlines()
output_file = open(pred_path, 'w', encoding = 'utf-8')
	

for err_sent in tqdm(input_file):
	pred_sent, pred_detail = correct(err_sent)
	output_file.write(pred_sent)

output_file.close()

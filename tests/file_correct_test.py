# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 
import sys
sys.path.append("../")
from pycorrector import corrector

in_file = sys.argv[1]
out_file = sys.argv[2]


def reader(in_file):
    lines = list()
    cout = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            text = line.split("\t")[0]
            lines.append(text)
            cout += 1
    print("in file: %s, cout: %d" % (in_file, cout))
    return lines


def saver(out_file, lines):
    cout = 0
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            f.write(line + '\n')
            cout += 1
    print("save file: %s, cout: %d" % (out_file, cout))


input_lines = reader(in_file)
correct_lines = list()
for line in input_lines:
    correct_sent, error_detail = corrector.correct(line)
    print("{}\t{}\t{}".format(
        line, correct_sent, error_detail))
    correct_lines.append(line + '\t' + correct_sent + '\t' + str(error_detail))
saver(out_file, correct_lines)

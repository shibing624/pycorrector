# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import csv

# 打开CSV文件
with open('ec_law_test.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)

    # 打开TSV文件
    with open('ec_law_test.tsv', 'w', newline='', encoding='utf-8') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')

        # 逐行读取CSV文件并写入TSV文件
        for row in csvreader:
            tsvwriter.writerow(row)
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# -----用户目录，存储模型文件-----
USER_DATA_DIR = os.path.expanduser('~/.pycorrector/datasets/')
os.makedirs(USER_DATA_DIR, exist_ok=True)
language_model_path = os.path.join(USER_DATA_DIR, 'zh_giga.no_cna_cmn.prune01244.klm')

# -----词典文件路径-----
# 通用分词词典文件  format: 词语 词频
word_freq_path = os.path.join(pwd_path, 'data/word_freq.txt')
# 中文常用字符集
common_char_path = os.path.join(pwd_path, 'data/common_char_set.txt')
# 同音字
same_pinyin_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
# 形似字
same_stroke_path = os.path.join(pwd_path, 'data/same_stroke.txt')
# 知名人名词典 format: 词语 词频
person_name_path = os.path.join(pwd_path, 'data/person_name.txt')
# 地名词典 format: 词语 词频
place_name_path = os.path.join(pwd_path, 'data/place_name.txt')
# 停用词
stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')
# 搭配词
ngram_words_path = os.path.join(pwd_path, 'data/ngram_words.txt')
# 英文拼写词频文件
en_dict_path = os.path.join(pwd_path, 'data/en/en.json.gz')

# -----深度模型文件路径 -----
# bert模型文件夹路径
bert_model_dir = os.path.join(USER_DATA_DIR, 'bert_models/chinese_finetuned_lm/')
os.makedirs(bert_model_dir, exist_ok=True)
# ernie模型文件夹路径: /Users/name/.paddle-ernie-cache/
# electra模型文件夹路径
electra_D_model_dir = os.path.join(USER_DATA_DIR, "electra_models/chinese_electra_base_discriminator_pytorch/")
electra_G_model_dir = os.path.join(USER_DATA_DIR, "electra_models/chinese_electra_base_generator_pytorch/")
# macbert模型文件路径
macbert_model_dir = os.path.join(USER_DATA_DIR, 'macbert_models/chinese_finetuned_correction/')
os.makedirs(macbert_model_dir, exist_ok=True)

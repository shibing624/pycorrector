# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys
sys.path.append("../")
from pycorrector import Corrector

from pycorrector.config import common_char_path, same_pinyin_path, \
    same_stroke_path, language_model_path, \
    word_freq_path, \
    custom_confusion_path, place_name_path, person_name_path, stopwords_path,custom_word_freq_path

# 使用三元文法语言模型（people_chars.klm）纠错效果更好：
# language_model_path = '../pycorrector/data/kenlm/people_chars.klm'
model = Corrector(common_char_path=common_char_path,
                  same_pinyin_path=same_pinyin_path,
                  same_stroke_path=same_stroke_path,
                  language_model_path=language_model_path,
                  word_freq_path=word_freq_path,
                  custom_word_freq_path=custom_word_freq_path,
                  custom_confusion_path=custom_confusion_path,
                  person_name_path=person_name_path,
                  place_name_path=place_name_path,
                  stopwords_path=stopwords_path
                  )

error_sentences = [
    '少先队员因该为老人让坐',
    '天地无垠大，我们的舞台无线大',
]
for line in error_sentences:
    correct_sent = model.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

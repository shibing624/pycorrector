# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from pycorrector import Corrector

from pycorrector.config import common_char_path, same_pinyin_path, \
    same_stroke_path, language_model_path, \
    word_freq_path, \
    custom_confusion_path


# language_model_path = '../pycorrector/data/test/people_chars.klm'
model = Corrector(common_char_path=common_char_path,
                  same_pinyin_path=same_pinyin_path,
                  same_stroke_path=same_stroke_path,
                  language_model_path=language_model_path,
                  word_freq_path=word_freq_path,
                  custom_confusion_path=custom_confusion_path)

error_sentences = [
    '少先队员因该为老人让坐',
    '天地无垠大，我们的舞台无线大',
]
for line in error_sentences:
    correct_sent = model.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

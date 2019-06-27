# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: pycorrector.api

from .config import common_char_path, same_pinyin_path, same_stroke_path, language_model_path, word_freq_path, \
    custom_confusion_path, custom_word_freq_path, place_name_path, person_name_path, stopwords_path
from .corrector import Corrector
from .utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin
from .utils.text_utils import traditional2simplified, simplified2traditional

__version__ = '0.1.7'

corrector = Corrector(common_char_path=common_char_path,
                      same_pinyin_path=same_pinyin_path,
                      same_stroke_path=same_stroke_path,
                      language_model_path=language_model_path,
                      word_freq_path=word_freq_path,
                      custom_word_freq_path=custom_word_freq_path,
                      custom_confusion_path=custom_confusion_path,
                      person_name_path=person_name_path,
                      place_name_path=place_name_path,
                      stopwords_path=stopwords_path)
get_same_pinyin = corrector.get_same_pinyin
get_same_stroke = corrector.get_same_stroke
set_custom_confusion_dict = corrector.set_custom_confusion_dict
set_custom_word = corrector.set_custom_word
set_language_model_path = corrector.set_language_model_path
correct = corrector.correct
ngram_score = corrector.ngram_score
ppl_score = corrector.ppl_score
word_frequency = corrector.word_frequency
detect = corrector.detect
enable_char_error = corrector.enable_char_error
enable_word_error = corrector.enable_word_error

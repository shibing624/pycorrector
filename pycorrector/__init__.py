# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: pycorrector.api

from .corrector import Corrector
from .config import common_char_path, same_pinyin_path, same_stroke_path, language_model_path, word_freq_path, \
    custom_confusion_path
from .utils.text_utils import traditional2simplified, simplified2traditional
from .utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin

__version__ = '0.1.5'

corrector = Corrector(common_char_path=common_char_path,
                      same_pinyin_path=same_pinyin_path,
                      same_stroke_path=same_stroke_path,
                      language_model_path=language_model_path,
                      word_freq_path=word_freq_path,
                      custom_confusion_path=custom_confusion_path)
get_same_pinyin = corrector.get_same_pinyin
get_same_stroke = corrector.get_same_stroke
set_custom_confusion_dict = corrector.set_custom_confusion_dict
set_language_model_path = corrector.set_language_model_path
correct = corrector.correct
ngram_score = corrector.ngram_score
ppl_score = corrector.ppl_score
word_frequency = corrector.word_frequency
detect = corrector.detect

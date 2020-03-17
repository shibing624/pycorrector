# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: init

from .config import language_model_path
from .corrector import Corrector
from .en_spell import en_correct
from .utils.logger import set_log_level
from .utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin, traditional2simplified, \
    simplified2traditional

corrector = Corrector()
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
set_log_level = set_log_level

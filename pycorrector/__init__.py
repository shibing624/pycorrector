# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: pycorrector.api

from .corrector import Corrector
from .detector import Detector
from .config import *

__version__ = '0.1.4'

corrector = Corrector(char_file_path=char_file_path,
                      same_pinyin_text_path=same_pinyin_text_path,
                      same_stroke_text_path=same_stroke_text_path,
                      language_model_path=language_model_path,
                      word_freq_path=word_freq_path)
get_same_pinyin = corrector.get_same_pinyin
get_same_stroke = corrector.get_same_stroke
correct = corrector.correct
ngram_score = corrector.detector.ngram_score
ppl_score = corrector.detector.ppl_score
word_frequency = corrector.detector.word_frequency
detect = corrector.detector.detect

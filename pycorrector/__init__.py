# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

from pycorrector.confusion_corrector import ConfusionCorrector
from pycorrector.corrector import Corrector
from pycorrector.detector import USER_DATA_DIR
from pycorrector.en_spell import EnSpell
from pycorrector.proper_corrector import ProperCorrector
from pycorrector.utils import text_utils, get_file, tokenizer, io_utils, math_utils
from pycorrector.utils.text_utils import (
    get_homophones_by_char,
    get_homophones_by_pinyin,
    traditional2simplified,
    simplified2traditional,
)
from pycorrector.version import __version__

# 中文纠错
corrector = Corrector()
get_same_pinyin = corrector.get_same_pinyin
get_same_stroke = corrector.get_same_stroke
set_custom_confusion_path_or_dict = corrector.set_custom_confusion_path_or_dict
set_custom_word_freq = corrector.set_custom_word_freq
set_language_model_path = corrector.set_language_model_path
correct = corrector.correct
ngram_score = corrector.ngram_score
ppl_score = corrector.ppl_score
word_frequency = corrector.word_frequency
detect = corrector.detect
enable_char_error = corrector.enable_char_error
enable_word_error = corrector.enable_word_error

# 英文纠错
en_spell = EnSpell()
en_correct = en_spell.correct
en_probability = en_spell.probability
set_en_custom_confusion_dict = en_spell.set_en_custom_confusion_dict

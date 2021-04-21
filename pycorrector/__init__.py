# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: init

from .bert import bert_corrector
from .bert.bert_corrector import BertCorrector
from .config import language_model_path
from .corrector import Corrector
from .en_spell import en_correct
from .ernie import ernie_corrector
from .ernie.ernie_corrector import ErnieCorrector
from .macbert import macbert_corrector
from .macbert.macbert_corrector import MacBertCorrector
from .utils import text_utils, get_file, tokenizer, io_utils, math_utils
from .utils.logger import set_log_level
from .utils.text_utils import get_homophones_by_char, get_homophones_by_pinyin, traditional2simplified, \
    simplified2traditional

ct = Corrector()
get_same_pinyin = ct.get_same_pinyin
get_same_stroke = ct.get_same_stroke
set_custom_confusion_dict = ct.set_custom_confusion_dict
set_custom_word_freq = ct.set_custom_word_freq
set_language_model_path = ct.set_language_model_path
correct = ct.correct
ngram_score = ct.ngram_score
ppl_score = ct.ppl_score
word_frequency = ct.word_frequency
detect = ct.detect
enable_char_error = ct.enable_char_error
enable_word_error = ct.enable_word_error

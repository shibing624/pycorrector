# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

from pycorrector.version import __version__  # noqa, isort:skip
from pycorrector.corrector import Corrector
from pycorrector.confusion_corrector import ConfusionCorrector
from pycorrector.deepcontext.deepcontext_corrector import DeepContextCorrector
from pycorrector.detector import Detector
from pycorrector.detector import USER_DATA_DIR
from pycorrector.en_spell_corrector import EnSpellCorrector
from pycorrector.ernie_csc.ernie_csc_corrector import ErnieCscCorrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector.proper_corrector import ProperCorrector
from pycorrector.seq2seq.conv_seq2seq_corrector import ConvSeq2SeqCorrector
from pycorrector.t5.t5_corrector import T5Corrector
from pycorrector.utils import text_utils, tokenizer, io_utils, math_utils, evaluate_utils
from pycorrector.utils.evaluate_utils import eval_model_batch
from pycorrector.utils.get_file import get_file
from pycorrector.utils.text_utils import (
    get_homophones_by_char,
    get_homophones_by_pinyin,
    traditional2simplified,
    simplified2traditional,
)

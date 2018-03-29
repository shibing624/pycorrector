# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Train seq2seq model for text grammar error correction

import math
import os
import shutil
import sys
import time
from collections import defaultdict
import numpy as np
import tensorflow as tf
from reader import EOS_ID
from fce_reader import FCEReader
from corrector_model import CorrectorModel

import seq2seq_config


def create_model(session, forward_only, model_path, config=seq2seq_config):
    """
    Create model and load parameters
    :param session:
    :param forward_only:
    :param model_path:
    :param config:
    :return:
    """
    model = CorrectorModel(
        config.max_vocabulary_size,
        config.max_vocabulary_size,
    )

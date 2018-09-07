# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import logging
import os
import pickle


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    handle.setFormatter(format)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.DEBUG)
    return logger


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if os.path.exists(pkl_path) and not overwrite:
        return
    with open(pkl_path, 'wb') as f:
        # pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(vocab, f, protocol=0)

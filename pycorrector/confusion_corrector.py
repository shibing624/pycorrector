# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 混淆集纠错，根据混淆集词典强制替换
功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
"""
import os

from loguru import logger


class ConfusionCorrector:
    def __init__(self, custom_confusion_path_or_dict=''):
        self.name = 'ConfusionCorrector'
        # 混淆集数据
        if isinstance(custom_confusion_path_or_dict, dict):
            self.custom_confusion = custom_confusion_path_or_dict
        elif isinstance(custom_confusion_path_or_dict, str):
            self.custom_confusion = self.load_custom_confusion_dict(custom_confusion_path_or_dict)
        else:
            raise ValueError('custom_confusion_path_or_dict must be dict or str.')
        logger.debug('Loaded confusion size: %d' % len(self.custom_confusion))

    @staticmethod
    def load_custom_confusion_dict(path):
        """
        加载自定义困惑集
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        confusion = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    info = line.split()
                    if len(info) < 2:
                        continue
                    error = info[0]
                    truth = info[1]
                    confusion[error] = truth
        else:
            logger.warning('file not found.%s' % path)
        return confusion

    def correct(self, sentence):
        """
        基于混淆集纠错
        :param sentence: str, 待纠错的文本
        :return: tuple(str, list), list(wrong, right, begin_idx, end_idx)
        """
        corrected_sentence = sentence
        details = []
        # 自定义混淆集加入疑似错误词典
        for err, truth in self.custom_confusion.items():
            idx = sentence.find(err)
            if idx > -1:
                corrected_sentence = sentence[:idx] + truth + sentence[(idx + len(err)):]
                maybe_err = [err, truth, idx, idx + len(err)]
                details.append(maybe_err)
        return corrected_sentence, details

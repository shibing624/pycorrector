# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import time
import os
from typing import List

sys.path.append('../..')
from pycorrector.utils.logger import logger
from pycorrector import config
from pycorrector.seq2seq.infer import Inference


class Seq2SeqCorrector(object):
    def __init__(
            self, model_dir=config.convseq2seq_model_dir,
            arch='convseq2seq', embed_size=128, hidden_size=128, dropout=0.25, max_length=128
    ):
        self.name = 'seq2seq_corrector'
        t1 = time.time()
        self.model = Inference(
            model_dir, arch,
            embed_size=embed_size, hidden_size=hidden_size,
            dropout=dropout, max_length=max_length
        )
        logger.debug('Loaded model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def seq2seq_correct(self, sentences: List[str]):
        """
        句子纠错
        :param: sentences, List[str]: 待纠错的句子
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        outputs = self.model.predict(sentences)
        return outputs


if __name__ == "__main__":
    m = Seq2SeqCorrector()
    error_sentences = [
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
        '他们的吵翻很不错，再说他们做的咖喱鸡也好吃',
        '影像小孩子想的快，学习管理的斑法',
        '餐厅的换经费产适合约会',
        '走路真的麻坊，我也没有喝的东西，在家汪了',
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '妈妈说："别趴地上了，快起来，你还吃饭吗？"，我说："好。"就扒起来了。',
        '你说：“怎么办？”我怎么知道？',
        '我父母们常常说：“那时候吃的东西太少，每天只能吃一顿饭。”想一想，人们都快要饿死，谁提出化肥和农药的污染。',
        '这本新书《居里夫人传》将的很生动有趣',
        '֍我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……֍新的食谱',
        '注意：“跨类保护”不等于“全类保护”。',
        '12.——对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性；',
        '《著作权法》（2020修正）第23条：“自然人的作品，其发表权、本法第',
        '三步检验法（三步检验标准）（three-step test）：若要',
    ]
    res = m.seq2seq_correct(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} , err:{}".format(sent, r[0], r[1]))

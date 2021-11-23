# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import sys

import torch
from transformers import BertTokenizer

sys.path.append('../..')

from pycorrector.macbert.macbert4csc import MacBert4Csc
from pycorrector.macbert.defaults import _C as cfg
from pycorrector.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(self, ckpt_path='output/macbert4csc/epoch=09-val_loss=0.01.ckpt',
                 vocab_path='output/macbert4csc/vocab.txt',
                 cfg_path='train_macbert4csc.yml'):
        logger.debug("device: {}".format(device))
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        cfg.merge_from_file(cfg_path)
        self.model = MacBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                      cfg=cfg,
                                                      map_location=device,
                                                      tokenizer=self.tokenizer)
        self.model.eval()
        self.model.to(device)
        logger.debug("device: {}".format(device))

    def predict(self, sentence_list):
        return self.model.predict(sentence_list)


if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    m = Inference(ckpt_path,
                  vocab_path='output/macbert4csc/vocab.txt',
                  cfg_path='train_macbert4csc.yml')
    inputs = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。',
        '你说：“怎么办？”我怎么知道？',
    ]
    outputs = m.predict(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b)
        print()

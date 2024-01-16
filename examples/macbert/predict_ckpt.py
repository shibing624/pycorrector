# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com), okcd00(okcd00@qq.com)
@description: 
"""
import argparse
import sys

import torch
from loguru import logger
from transformers import BertTokenizerFast

sys.path.append('../..')

from pycorrector.macbert.macbert4csc import MacBert4Csc
from pycorrector.macbert.softmaskedbert4csc import SoftMaskedBert4Csc
from pycorrector.macbert.defaults import _C as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(
            self,
            ckpt_path='outputs-macbert4csc/epoch=09-val_loss=0.01.ckpt',
            vocab_dir='outputs-macbert4csc/',
            cfg_path='train_macbert4csc.yml'
    ):
        logger.debug("device: {}".format(device))
        self.tokenizer = BertTokenizerFast.from_pretrained(vocab_dir)
        cfg.merge_from_file(cfg_path)

        if 'macbert4csc' in cfg_path:
            self.model = MacBert4Csc.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=cfg,
                map_location=device,
                tokenizer=self.tokenizer
            )
        elif 'softmaskedbert4csc' in cfg_path:
            self.model = SoftMaskedBert4Csc.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=cfg,
                map_location=device,
                tokenizer=self.tokenizer
            )
        else:
            raise ValueError("model not found.")
        self.model.to(device)
        self.model.eval()

    def predict(self, sentences):
        """
        文本纠错模型预测
        Args:
            sentences: list
                输入文本列表
        Returns: tuple
            corrected_texts(list)
        """
        return self.model.predict(sentences)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--ckpt_path", default="outputs-macbert4csc/epoch=09-val_loss=0.01.ckpt",
                        help="path to ckpt file", type=str)
    parser.add_argument("--vocab_dir", default="outputs-macbert4csc/", help="path to vocab file", type=str)
    parser.add_argument("--config_file", default="train_macbert4csc.yml", help="path to config file", type=str)
    args = parser.parse_args()
    m = Inference(args.ckpt_path, args.vocab_dir, args.config_file)
    inputs = [
        '它的本领是呼风唤雨，因此能灭火防灾。狎鱼后面是獬豸。獬豸通常头上长着独角，有时又被称为独角羊。它很聪彗，而且明辨是非，象征着大公无私，又能镇压斜恶。',
        '老是较书。',
        '少先队 员因该 为老人让 坐',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    outputs = m.predict(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b)
        print()

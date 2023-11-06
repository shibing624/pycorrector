# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: train seq2seq model

# #### PyTorch代码
# - [seq2seq-tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# - [Tutorial from Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
# - [IBM seq2seq](https://github.com/IBM/pytorch-seq2seq)
# - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
# - [text-generation](https://github.com/shibing624/text-generation)
"""

import argparse
import sys

from loguru import logger

sys.path.append('../..')
from pycorrector.seq2seq.conv_seq2seq_model import ConvSeq2SeqModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/sighan_2015/train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="../data/sighan_2015/test.tsv", type=str, help="Test file")
    parser.add_argument("--do_train", action="store_true", help="Whether not to train")
    parser.add_argument("--do_predict", action="store_true", help="Whether not to predict")
    parser.add_argument("--output_dir", default="outputs-sighan-convseq2seq/", type=str, help="Dir for model save.")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--num_epochs", default=200, type=int, help="Epoch num.")
    args = parser.parse_args()
    logger.info(args)

    # Train model with train data file
    if args.do_train:
        logger.info('Loading data...')
        model = ConvSeq2SeqModel(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            model_dir=args.output_dir,
            max_length=args.max_length
        )
        model.train_model(args.train_file)
        model.eval_model(args.test_file)

    if args.do_predict:
        model = ConvSeq2SeqModel(
            model_dir=args.output_dir,
            max_length=args.max_length
        )
        sentences = [
            '老是较书。',
            '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
            '遇到一位很棒的奴生跟我聊天。',
        ]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences))


if __name__ == '__main__':
    main()

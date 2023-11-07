# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys

sys.path.append("../..")
from pycorrector.deepcontext.deepcontext_model import DeepContextModel
from pycorrector.deepcontext.deepcontext_corrector import DeepContextCorrector


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--train_path", default="../data/wiki_zh_200.txt", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--do_train", action="store_true", help="Whether not to train")
    parser.add_argument("--do_predict", action="store_true", help="Whether not to predict")
    parser.add_argument("--output_dir", default="outputs-deepcontext-lm/", type=str, help="Dir for model save.")
    # Other parameters
    parser.add_argument("--max_length", default=1024, type=int, help="Max length of input sentence.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
    parser.add_argument("--min_freq", default=1, type=int, help="Mini word frequency.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--num_epochs", default=80, type=int, help="Epoch num.")
    args = parser.parse_args()
    print(args)

    # Train
    if args.do_train:
        m = DeepContextModel(args.output_dir, max_length=args.max_length)
        # Train model with train data file
        m.train_model(
            args.train_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            min_freq=args.min_freq,
            dropout=args.dropout
        )
        sent = '老是较书。'
        pred_words_res = m.predict_mask_token(list(sent), mask_index=2)
        print(sent, pred_words_res)
    # Predict
    if args.do_predict:
        m = DeepContextCorrector(args.output_dir, max_length=args.max_length)
        inputs = [
            '老是较书。',
            '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
            '遇到一位很棒的奴生跟我聊天。',
            '遇到一位很美的女生跟我疗天。',
            '他们只能有两个选择：接受降新或自动离职。',
            '王天华开心得一直说话。'
        ]
        for i in inputs:
            output = m.correct(i)
            print(output)
            print()


if __name__ == "__main__":
    main()

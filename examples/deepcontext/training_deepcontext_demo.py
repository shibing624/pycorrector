# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import os
import sys

sys.path.append("../..")
from pycorrector.deepcontext.train import train
from pycorrector.deepcontext.infer import Inference
from pycorrector.deepcontext.preprocess import parse_xml_file, save_corpus_data, get_data_file


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--raw_train_path",
                        default="../pycorrector/data/cn/sighan_2015/train.tsv", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
                        )
    parser.add_argument("--dataset", default="sighan", type=str,
                        help="Dataset name. selected in the list:" + ", ".join(["sighan", "cged"])
                        )
    parser.add_argument("--no_segment", action="store_true", help="Whether not to segment train data in preprocess")
    parser.add_argument("--do_train", action="store_true", help="Whether not to train")
    parser.add_argument("--do_predict", action="store_true", help="Whether not to predict")
    parser.add_argument("--segment_type", default="char", type=str,
                        help="Segment data type, selected in list: " + ", ".join(["char", "word"]))
    parser.add_argument("--model_dir", default="output/models/", type=str, help="Dir for model save.")
    parser.add_argument("--train_path", default="output/train.txt", type=str, help="Train file after preprocess.")
    parser.add_argument("--vocab_path", default="output/vocab.txt", type=str, help="Vocab file for train data.")

    # Other parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--embed_size", default=128, type=int, help="Embedding size.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--n_layers", default=2, type=int, help="Num layers.")
    parser.add_argument("--min_freq", default=1, type=int, help="Mini word frequency.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=20, type=int, help="Epoch num.")

    args = parser.parse_args()
    print(args)

    # Preprocess
    os.makedirs(args.model_dir, exist_ok=True)

    # Train
    if args.do_train:
        # Preprocess
        args.use_segment = False if args.no_segment else True
        data_list = []
        if args.dataset == 'sighan':
            data_list.extend(get_data_file(args.raw_train_path, args.use_segment, args.segment_type))
        else:
            data_list.extend(parse_xml_file(args.raw_train_path, args.use_segment, args.segment_type))
        save_corpus_data(data_list, args.train_path)

        # Train model with train data file
        train(args.train_path,
              args.model_dir,
              args.vocab_path,
              batch_size=args.batch_size,
              epochs=args.epochs,
              word_embed_size=args.embed_size,
              hidden_size=args.hidden_size,
              learning_rate=args.learning_rate,
              n_layers=args.n_layers,
              min_freq=args.min_freq,
              dropout=args.dropout
              )

    # Predict
    if args.do_predict:
        inference = Inference(args.model_dir, args.vocab_path)
        inputs = [
            '老是较书。',
            '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
            '遇到一位很棒的奴生跟我聊天。',
            '遇到一位很美的女生跟我疗天。',
            '他们只能有两个选择：接受降新或自动离职。',
            '王天华开心得一直说话。'
        ]
        for i in inputs:
            output = inference.predict(i)
            print('input  :', i)
            print('predict:', output)
            print()


if __name__ == "__main__":
    main()

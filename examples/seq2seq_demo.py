# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys

sys.path.append("..")
from pycorrector.seq2seq.train import train
from pycorrector.seq2seq.infer import Inference
from pycorrector.seq2seq.preprocess import get_data_file, parse_xml_file, save_corpus_data


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--raw_train_path",
                        default="../pycorrector/data/cn/sighan_2015.tsv",
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
                        )
    parser.add_argument("--dataset", default="sighan", type=str,
                        help="Dataset name. selected in the list:" + ", ".join(["sighan", "cged"])
                        )
    parser.add_argument("--no_segment", action="store_true", help="Whether not to segment train data in preprocess")
    parser.add_argument("--segment_type", default="char", type=str,
                        help="Segment data type, selected in list: " + ", ".join(["char", "word"]))
    parser.add_argument("--model_name_or_path",
                        default="bert-base-chinese",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models",
                        )
    parser.add_argument("--model_dir", default="output/bertseq2seq/", type=str, help="Dir for model save.")
    parser.add_argument("--arch",
                        default="bertseq2seq",
                        type=str,
                        required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(
                            ['seq2seq', 'convseq2seq', 'bertseq2seq']),
                        )
    parser.add_argument("--train_path", default="output/train.txt", type=str, help="Train file after preprocess.")
    parser.add_argument("--test_path", default="output/test.txt", type=str, help="Test file after preprocess.")
    parser.add_argument("--src_vocab_path", default="output/vocab_source.txt", type=str, help="Vocab file for src.")
    parser.add_argument("--trg_vocab_path", default="output/vocab_target.txt", type=str, help="Vocab file for trg.")

    # Other parameters
    parser.add_argument("--max_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, sequences shorter padded.",
                        )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--embed_size", default=128, type=int, help="Embedding size.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=40, type=int, help="Epoch num.")

    args = parser.parse_args()
    print(args)

    # Preprocess
    os.makedirs(args.model_dir, exist_ok=True)
    args.use_segment = False if args.no_segment else True
    data_list = []
    if args.dataset == 'sighan':
        data_list.extend(get_data_file(args.raw_train_path, args.use_segment, args.segment_type))
    else:
        data_list.extend(parse_xml_file(args.raw_train_path, args.use_segment, args.segment_type))
    save_corpus_data(data_list, args.train_path, args.test_path)

    # Train
    train(args.arch,
          args.train_path,
          args.batch_size,
          args.embed_size,
          args.hidden_size,
          args.dropout,
          args.epochs,
          args.src_vocab_path,
          args.trg_vocab_path,
          args.model_dir,
          args.max_length,
          args.use_segment
          )

    # Predict
    inference = Inference(args.arch,
                          args.model_dir,
                          args.src_vocab_path,
                          args.trg_vocab_path,
                          embed_size=args.embed_size,
                          hidden_size=args.hidden_size,
                          dropout=args.dropout,
                          max_length=args.max_length,
                          )
    inputs = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    outputs = inference.predict(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b)
        print()


if __name__ == "__main__":
    main()

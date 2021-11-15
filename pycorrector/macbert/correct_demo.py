# -*- coding: utf-8 -*-
"""
@Time   :   2021-02-03 21:57:15
@File   :   correct_demo.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import sys

sys.path.append('../..')
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector import config


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--macbert_model_dir", default=config.macbert_model_dir,
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()

    nlp = MacBertCorrector(args.macbert_model_dir).macbert_correct

    i = nlp('今天新情很好')
    print(i)

    i = nlp('少先队员英该为老人让座')
    print(i)

    i = nlp('机器学习是人工智能领遇最能体现智能的一个分知。')
    print(i)

    i = nlp('机其学习是人工智能领遇最能体现智能的一个分知。')
    print(i)

    print(nlp('老是较书。'))
    print(nlp('遇到一位很棒的奴生跟我聊天。'))


if __name__ == "__main__":
    main()

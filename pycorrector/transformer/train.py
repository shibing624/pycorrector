# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

import opennmt as onmt

from pycorrector.transformer import config
from pycorrector.transformer.corpus_reader import CGEDReader, load_word_dict, save_word_dict
from pycorrector.transformer.model import source_inputter, target_inputter, train


def main(model_dir='',
         src_train_path='',
         tgt_train_path='',
         vocab_path='',
         maximum_length=100,
         shuffle_buffer_size=1000000,
         gradients_accum=8,
         train_steps=10000,
         save_every=1000,
         report_every=50):
    data_reader = CGEDReader(src_train_path)
    src_input_texts = data_reader.build_dataset(src_train_path)
    tgt_input_texts = data_reader.build_dataset(tgt_train_path)

    # load or save word dict
    if os.path.exists(vocab_path):
        char2id = load_word_dict(vocab_path)
        id2char = {int(j): i for i, j in char2id.items()}
        chars = set([i for i in char2id.keys()])
    else:
        print('Training data...')
        print('input_texts:', src_input_texts[0])
        print('target_texts:', tgt_input_texts[0])
        max_input_texts_len = max([len(text) for text in src_input_texts])

        print('num of samples:', len(src_input_texts))
        print('max sequence length for inputs:', max_input_texts_len)

        chars = data_reader.read_vocab(src_input_texts + tgt_input_texts)
        id2char = {i: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        save_word_dict(char2id, vocab_path)

    inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": vocab_path,
        "target_vocabulary": vocab_path
    })
    # opennmt train model
    train(model_dir,
          inputter,
          src_train_path,
          tgt_train_path,
          maximum_length=maximum_length,
          shuffle_buffer_size=shuffle_buffer_size,
          gradients_accum=gradients_accum,
          train_steps=train_steps,
          save_every=save_every,
          report_every=report_every)


if __name__ == "__main__":
    main(config.model_dir,
         src_train_path=config.src_train_path,
         tgt_train_path=config.tgt_train_path,
         vocab_path=config.vocab_path,
         maximum_length=config.maximum_length,
         shuffle_buffer_size=config.shuffle_buffer_size,
         gradients_accum=config.gradients_accum,
         train_steps=config.train_steps,
         save_every=config.save_every,
         report_every=config.report_every)

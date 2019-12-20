# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import sys

import tensorflow as tf

sys.path.append('../..')

from pycorrector.transformer import config
from pycorrector.transformer.corpus_reader import CGEDReader, save_word_dict
from pycorrector.transformer.model import train, model, checkpoint


def main(model_dir='',
         src_train_path='',
         tgt_train_path='',
         src_vocab_path='',
         tgt_vocab_path='',
         batch_size=3072,
         maximum_length=100,
         train_steps=10000,
         save_every=1000,
         report_every=50):
    data_reader = CGEDReader(src_train_path)
    src_input_texts = data_reader.build_dataset(src_train_path)
    tgt_input_texts = data_reader.build_dataset(tgt_train_path)

    # load or save word dict
    if not os.path.exists(src_vocab_path):
        print('Training data...')
        print('input_texts:', src_input_texts[0])
        print('target_texts:', tgt_input_texts[0])
        max_input_texts_len = max([len(text) for text in src_input_texts])

        print('num of samples:', len(src_input_texts))
        print('max sequence length for inputs:', max_input_texts_len)

        src_vocab = data_reader.read_vocab(src_input_texts)
        id2char = {i: j for i, j in enumerate(src_vocab)}
        char2id = {j: i for i, j in id2char.items()}
        save_word_dict(char2id, src_vocab_path)

        tgt_vocab = data_reader.read_vocab(tgt_input_texts)
        id2char = {i: j for i, j in enumerate(tgt_vocab)}
        char2id = {j: i for i, j in id2char.items()}
        save_word_dict(char2id, tgt_vocab_path)

    data_config = {
        "source_vocabulary": src_vocab_path,
        "target_vocabulary": tgt_vocab_path
    }

    model.initialize(data_config)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=5)
    if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    train(src_train_path, tgt_train_path, checkpoint_manager,
          batch_size=batch_size,
          maximum_length=maximum_length,
          train_steps=train_steps,
          save_every=save_every,
          report_every=report_every)


if __name__ == "__main__":
    if config.gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    main(config.model_dir,
         src_train_path=config.src_train_path,
         tgt_train_path=config.tgt_train_path,
         src_vocab_path=config.src_vocab_path,
         tgt_vocab_path=config.tgt_vocab_path,
         batch_size=config.batch_size,
         maximum_length=config.maximum_length,
         train_steps=config.train_steps,
         save_every=config.save_every,
         report_every=config.report_every)

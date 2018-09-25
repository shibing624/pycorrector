# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Train seq2seq model for text grammar error correction
import os
import sys
sys.path.append('../..')
import numpy as np
from keras.models import load_model
from seq2seq.models import AttentionSeq2Seq
from pycorrector.seq2seq import cged_config as config
from pycorrector.seq2seq.corpus_reader import CGEDReader, save_word_dict, load_word_dict
from pycorrector.seq2seq.seq2seq_model import create_model, callback, evaluate
from pycorrector.utils.io_utils import get_logger

logger = get_logger(__name__)


def train(train_path=None,
          save_model_path=None,
          encoder_model_path=None,
          decoder_model_path=None,
          save_input_token_path=None,
          save_target_token_path=None,
          batch_size=64,
          epochs=10,
          rnn_hidden_dim=200):
    print('Training model...')
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path)
    print('input_texts:', input_texts[0])
    print('target_texts:', target_texts[0])
    max_input_texts_len = max([len(text) for text in input_texts])
    max_target_texts_len = max([len(text) for text in target_texts])

    print('num of samples:', len(input_texts))
    print('max sequence length for inputs:', max_input_texts_len)
    print('max sequence length for outputs:', max_target_texts_len)

    # load or save word dict
    if os.path.exists(save_input_token_path) and os.path.exists(save_target_token_path):
        input_token_index = load_word_dict(save_input_token_path)
        target_token_index = load_word_dict(save_target_token_path)
    else:
        input_characters = data_reader.read_vocab(input_texts)
        target_characters = data_reader.read_vocab(target_texts)
        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        save_word_dict(input_token_index, save_input_token_path)
        save_word_dict(target_token_index, save_target_token_path)

    encoder_input_data = np.zeros((len(input_texts), max_input_texts_len, len(input_token_index)), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_target_texts_len, len(target_token_index)), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_target_texts_len, len(target_token_index)), dtype='float32')

    # one hot representation
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            if char in input_token_index:
                encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is a head of decoder_input_data by one timestep
            if char in target_token_index:
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    logger.info("Data loaded.")

    # model
    logger.info("Training seq2seq model...")
    if os.path.exists(save_model_path) and os.path.exists(encoder_model_path):
        model = load_model(save_model_path)
        encoder_model = load_model(encoder_model_path)
        decoder_model = load_model(decoder_model_path)
    else:
        model = AttentionSeq2Seq(input_dim=len(input_token_index), input_length=max_input_texts_len,
                                 hidden_dim=rnn_hidden_dim, output_length=max_target_texts_len,
                                 output_dim=len(target_token_index), depth=4)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # Run training
    callbacks_list = callback(save_model_path, logger)
    model.fit(x=[encoder_input_data, decoder_input_data],
              y=decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              callbacks=callbacks_list)
    encoder_model.save(encoder_model_path)
    decoder_model.save(decoder_model_path)
    logger.info("Model save to " + save_model_path)
    logger.info("Training has finished.")

    evaluate(encoder_model, decoder_model, len(input_token_index),
             len(target_token_index), rnn_hidden_dim, target_token_index,
             max_target_texts_len, encoder_input_data, input_texts)


if __name__ == "__main__":
    train(train_path=config.train_path,
          save_model_path=config.save_model_path,
          encoder_model_path=config.encoder_model_path,
          decoder_model_path=config.decoder_model_path,
          save_input_token_path=config.input_vocab_path,
          save_target_token_path=config.target_vocab_path,
          batch_size=config.batch_size,
          epochs=config.epochs,
          rnn_hidden_dim=config.rnn_hidden_dim)

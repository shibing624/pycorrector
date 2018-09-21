# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import numpy as np
import tensorflow as tf
from keras.models import load_model

from pycorrector.seq2seq import cged_config as config
from pycorrector.seq2seq.corpus_reader import CGEDReader, load_word_dict
from pycorrector.seq2seq.reader import EOS_TOKEN, GO_TOKEN
from pycorrector.utils.io_utils import get_logger

logger = get_logger(__name__)


class Infer(object):
    def __init__(self, config=None):
        train_path = config.train_path
        encoder_model_path = config.encoder_model_path
        decoder_model_path = config.decoder_model_path
        save_input_token_path = config.input_vocab_path
        save_target_token_path = config.target_vocab_path

        # load dict
        self.input_token_index = load_word_dict(save_input_token_path)
        self.target_token_index = load_word_dict(save_target_token_path)

        data_reader = CGEDReader(train_path)
        input_texts, target_texts = data_reader.build_dataset(train_path)
        self.max_input_texts_len = max([len(text) for text in input_texts])
        self.max_target_texts_len = max([len(text) for text in target_texts])
        logger.info("Data loaded.")

        # load model
        self.encoder_model = load_model(encoder_model_path)
        self.decoder_model = load_model(decoder_model_path)
        logger.info("Loaded seq2seq model.")
        self.graph = tf.get_default_graph()

    def _decode_sequence(self, encoder_input_data):
        decoded_sentence = ''
        with self.graph.as_default():
            # Encode the input as state vectors.
            states_value = self.encoder_model.predict(encoder_input_data)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, len(self.target_token_index)))
            # Populate the first character of target sequence with the start character.
            # first_char = encoder_input_data[0]
            target_seq[0, 0, self.target_token_index[GO_TOKEN]] = 1.0

            reverse_target_char_index = dict(
                (i, char) for char, i in self.target_token_index.items())

            for _ in range(self.max_target_texts_len):
                output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == EOS_TOKEN:
                    break
                decoded_sentence += sampled_char
                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, len(self.target_token_index)))
                target_seq[0, 0, sampled_token_index] = 1.0
                # Update states
                states_value = [h, c]
        return decoded_sentence

    def infer(self, input_text):
        encoder_input_data = np.zeros((1, self.max_input_texts_len, len(self.input_token_index)),
                                      dtype='float32')
        # one hot representation
        for i, char in enumerate(input_text):
            if char in self.input_token_index:
                encoder_input_data[0, i, self.input_token_index[char]] = 1.0
        # Take one sequence decoding.
        decoded_sentence = self._decode_sequence(encoder_input_data)
        logger.info('Input sentence:%s' % input_text)
        logger.info('Decoded sentence:%s' % decoded_sentence)


if __name__ == "__main__":
    inference = Infer(config=config)
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
    ]
    for i in inputs:
        inference.infer(i)

    while True:
        input_str = input('input your string:')
        inference.infer(input_str)

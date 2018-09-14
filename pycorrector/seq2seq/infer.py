# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import numpy as np
from keras.layers import Input, LSTM
from keras.models import load_model

from pycorrector.seq2seq import cged_config as config
from pycorrector.seq2seq.corpus_reader import CGEDReader, load_word_dict
from pycorrector.seq2seq.reader import EOS_TOKEN, GO_TOKEN
from pycorrector.utils.io_utils import get_logger

logger = get_logger(__name__)


def evaluate(encoder_model, decoder_model, num_encoder_tokens,
             num_decoder_tokens, rnn_hidden_dim, target_token_index,
             max_decoder_seq_length, encoder_input_data, input_texts):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(rnn_hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(rnn_hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    def decode_seq(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        first_char = GO_TOKEN
        target_seq[0, 0, target_token_index[first_char]] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            if sampled_char != EOS_TOKEN:
                decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == EOS_TOKEN or
                        len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    for seq_index in range(10):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_seq(input_seq)

        logger.info('Input sentence:%s' % input_texts[seq_index])
        logger.info('Decoded sentence:%s' % decoded_sentence)


def decode_sequence(encoder_model, decoder_model,
                    num_decoder_tokens, target_token_index,
                    encoder_input_data, max_target_texts_len):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(encoder_input_data)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # first_char = encoder_input_data[0]
    target_seq[0, 0, target_token_index[GO_TOKEN]] = 1.0

    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    decoded_sentence = ''
    for _ in range(max_target_texts_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == EOS_TOKEN:
            break
        decoded_sentence += sampled_char
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        # Update states
        states_value = [h, c]
    return decoded_sentence


def infer(input_text):
    # one hot representation
    for i, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input_data[0, i, input_token_index[char]] = 1.0
    # Take one sequence decoding.
    decoded_sentence = decode_sequence(encoder_model, decoder_model,
                                       len(target_token_index), target_token_index,
                                       encoder_input_data, max_target_texts_len)
    logger.info('Input sentence:%s' % input_text)
    logger.info('Decoded sentence:%s' % decoded_sentence)


if __name__ == "__main__":
    train_path = config.train_path
    encoder_model_path = config.encoder_model_path
    decoder_model_path = config.decoder_model_path
    save_input_token_path = config.input_vocab_path
    save_target_token_path = config.target_vocab_path

    # load dict
    input_token_index = load_word_dict(save_input_token_path)
    target_token_index = load_word_dict(save_target_token_path)

    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path)
    max_input_texts_len = max([len(text) for text in input_texts])
    max_target_texts_len = max([len(text) for text in target_texts])
    encoder_input_data = np.zeros((1, max_input_texts_len, len(input_token_index)), dtype='float32')
    logger.info("Data loaded.")

    # load model
    encoder_model = load_model(encoder_model_path)
    decoder_model = load_model(decoder_model_path)
    logger.info("Loaded seq2seq model.")

    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '不能人类实现更美好的将来。',
    ]
    for i in inputs:
        infer(i)

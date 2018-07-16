# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 


import numpy as np
from keras.layers import Input
from keras.models import Model, load_model

from pycorrector.seq2seq import cged_config as config
from pycorrector.seq2seq.corpus_reader import GO_TOKEN
from pycorrector.seq2seq.corpus_reader import CGEDReader
from pycorrector.utils.io_utils import get_logger

from pycorrector.seq2seq.reader import EOS_TOKEN

logger = get_logger(__name__)


def decode_sequence(model, rnn_hidden_dim,input_token_index,
                    num_decoder_tokens, target_token_index,encoder_input_data,
                    reverse_target_char_index, max_decoder_seq_length):
    # construct the encoder and decoder
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(rnn_hidden_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(rnn_hidden_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    # Encode the input as state vectors.
    states_value = encoder_model.predict(encoder_input_data)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index[first_char]] = 1.

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


def infer(train_path=None,
          test_path=None,
          save_model_path=None,
          rnn_hidden_dim=200):
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(test_path)

    input_characters = data_reader.read_vocab(input_texts)
    target_characters = data_reader.read_vocab(target_texts)
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_len = max([len(text) for text in input_texts])
    max_decoder_seq_len = max([len(text) for text in target_texts])

    print('num of samples:', len(input_texts))
    print('num of unique input tokens:', num_encoder_tokens)
    print('num of unique output tokens:', num_decoder_tokens)
    print('max sequence length for inputs:', max_encoder_seq_len)
    print('max sequence length for outputs:', max_decoder_seq_len)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_len, num_encoder_tokens), dtype='float32')

    # one hot representation
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
    logger.info("Data loaded.")

    # model
    logger.info("Infer seq2seq model...")
    model = load_model(save_model_path)

    decoded_sentences = decode_sequence(model, encoder_input_data, )
    for seq_index in input_text:
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentences[seq_index])

    logger.info("Infer has finished.")


if __name__ == "__main__":
    infer(train_path=config.train_path,
          test_path=config.test_path,
          save_model_path=config.save_model_path,
          rnn_hidden_dim=config.rnn_hidden_dim)

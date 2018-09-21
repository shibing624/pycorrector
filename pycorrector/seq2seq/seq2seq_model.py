# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: seq2seq model with keras (refs: keras-example)
import numpy as np
from keras import Input
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense
from keras.models import Model

from pycorrector.seq2seq.infer import logger
from pycorrector.seq2seq.reader import GO_TOKEN, EOS_TOKEN


def create_model(num_encoder_tokens, num_decoder_tokens, rnn_hidden_dim=200):
    # 1.Define encoder model
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
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # 2.Define inference encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
    # 3.Define inference decoder model
    decoder_state_input_h = Input(shape=(rnn_hidden_dim,))
    decoder_state_input_c = Input(shape=(rnn_hidden_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def callback(save_model_path, logger=None):
    # Print the batch number at the beginning of every batch.
    if logger:
        batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: logger.info('batch: %d' % batch))
    # define the checkpoint, save model
    checkpoint = ModelCheckpoint(save_model_path)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
    return [batch_print_callback, checkpoint, early_stop]


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
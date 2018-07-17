# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kenlm

import numpy as np
import tensorflow as tf

from pycorrector.rnn_attention.nlc_data import *
from pycorrector.rnn_attention.nlc_model import NLCModel
from pycorrector.rnn_attention.util import get_tokenizer

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 400, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "./output", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./output/ckpt", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_string("lmfile", '../data/kenlm/people_chars.klm', "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0.3, "Language model relative weight.")

FLAGS = tf.app.flags.FLAGS
reverse_vocab, vocab = None, None
lm = None


def create_model(session, vocab_size, forward_only):
    model = NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only)
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.train_dir)
    print("checkpoint file", checkpoint_file)
    if checkpoint_file:
        print("Reading model parameters from %s" % checkpoint_file)
        model.saver.restore(session, checkpoint_file)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def tokenize(sent, vocab, depth=FLAGS.num_layers):
    align = pow(2, depth - 1)
    token_ids = sentence_to_token_ids(sent, vocab, get_tokenizer())
    ones = [1] * len(token_ids)
    pad = (align - len(token_ids)) % align

    token_ids += [PAD_ID] * pad
    ones += [0] * pad

    source = np.array(token_ids).reshape([-1, 1])
    mask = np.array(ones).reshape([-1, 1])

    return source, mask


def detokenize(sents, reverse_vocab):
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent
    a = [detok_sent(s) for s in sents]
    return [detok_sent(s) for s in sents]


def lm_rank(strs, probs):
    if lm is None:
        return ''.join(strs)
    a = FLAGS.alpha
    lmscores = [lm.score(s) / (1 + len(s.split())) for s in strs]
    probs = [p / (len(s) + 1) for (s, p) in zip(strs, probs)]
    for (s, p, l) in zip(strs, probs, lmscores):
        print(s, p, l)

    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    return generated  # , score, nw_score, lm_score


def decode_beam(model, sess, encoder_output, max_beam_size):
    toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
    return toks.tolist(), probs.tolist()


def fix_sent(model, sess, sent):
    # Tokenize
    input_toks, mask = tokenize(sent, vocab)
    # Encode
    encoder_output = model.encode(sess, input_toks, mask)
    # Decode
    beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size)
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)
    # Language Model ranking
    best_str = lm_rank(beam_strs, probs)
    # Return
    return best_str


def decode():
    # Prepare NLC data.
    global reverse_vocab, vocab, lm

    if FLAGS.lmfile is not None:
        print("Loading Language model from %s" % FLAGS.lmfile)
        # lm = kenlm.LanguageModel(FLAGS.lmfile)
        lm = kenlm.Model(FLAGS.lmfile)
    else:
        print('No lmfile, better to add kenlm arpa data file')

    print("Preparing NLC data in %s" % FLAGS.data_dir)

    x_train, y_train, x_dev, y_dev, vocab_path = prepare_nlc_data(
        FLAGS.data_dir, FLAGS.max_vocab_size, tokenizer=get_tokenizer())
    vocab, reverse_vocab = initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab_size, False)

        while True:
            sent = input("Enter a sentence: ")

            output_sent = fix_sent(model, sess, sent)

            print("Candidate: ", output_sent)


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()

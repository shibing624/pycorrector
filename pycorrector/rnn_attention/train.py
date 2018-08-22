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

import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

from pycorrector.rnn_attention.nlc_data import prepare_nlc_data, initialize_vocabulary
from pycorrector.rnn_attention.nlc_model import NLCModel
from pycorrector.rnn_attention.util import get_tokenizer
from pycorrector.rnn_attention.util import pair_iter

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 6, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 400, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "./output", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./output/ckpt", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")

FLAGS = tf.app.flags.FLAGS


def create_model(session, vocab_size, forward_only):
    model = NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only, optimizer=FLAGS.optimizer)
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.train_dir)
    if checkpoint_file:
        logging.info("Reading model parameters from %s" % checkpoint_file)
        model.saver.restore(session, checkpoint_file)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def validate(model, sess, x_dev, y_dev):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, FLAGS.batch_size,
                                                                            FLAGS.num_layers):
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Preparing NLC data in %s" % FLAGS.data_dir)
    x_train, y_train, x_dev, y_dev, vocab_path = prepare_nlc_data(
        FLAGS.data_dir, FLAGS.max_vocab_size, tokenizer=get_tokenizer())

    vocab, _ = initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    logging.info("Vocabulary size: %d" % vocab_size)

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))

    with tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab_size, False)

        logging.info('Initial validation cost: %f' % validate(model, sess, x_dev, y_dev))

        epoch = 0
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        while FLAGS.epochs == 0 or epoch < FLAGS.epochs:
            epoch += 1
            current_step = 0

            # Train
            epoch_tic = time.time()
            for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS.batch_size,
                                                                                    FLAGS.num_layers):
                # Get a batch and make a step.
                grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)
                total_iters += np.sum(target_mask)
                tps = total_iters / (time.time() - start_time)
                current_step += 1

                lengths = np.sum(target_mask, axis=0)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)

                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * cost
                    exp_length = 0.99 * exp_length + 0.01 * mean_length
                    exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                cost = cost / mean_length

                if current_step % FLAGS.print_every == 0:
                    logging.info('epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, tps %f, '
                                 'length mean/std %f/%f' % (epoch, current_step, cost, exp_cost / exp_length, grad_norm,
                                                            param_norm, tps, mean_length, std_length))
            epoch_toc = time.time()

            # Checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")

            # Validate
            valid_cost = validate(model, sess, x_dev, y_dev)

            logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

            if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                sess.run(model.learning_rate_decay_op)
                model.saver.restore(sess, checkpoint_path)
            else:
                previous_losses.append(valid_cost)
                model.saver.save(sess, checkpoint_path)
            sys.stdout.flush()


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()

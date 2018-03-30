# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Train seq2seq model for text grammar error correction

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

import seq2seq_config
from corrector_model import CorrectorModel
from fce_reader import FCEReader
from tf_util import get_ckpt_path


def create_model(session, forward_only, model_path, config=seq2seq_config):
    """
    Create model and load parameters
    :param session:
    :param forward_only:
    :param model_path:
    :param config:
    :return:
    """
    model = CorrectorModel(
        config.max_vocab_size,
        config.max_vocab_size,
        config.buckets,
        config.size,
        config.num_layers,
        config.max_gradient_norm,
        config.batch_size,
        config.learning_rate,
        config.learning_rate_decay_factor,
        config.use_lstm,
        forward_only=forward_only,
        config=config)
    ckpt_path = get_ckpt_path(model_path)
    if ckpt_path:
        print("Read model parameters from %s" % ckpt_path)
        model.saver.restore(session, ckpt_path)
    else:
        print('Create model...')
        session.run(tf.global_variables_initializer())
    return model


def train(data_reader, train_path, test_path, model_path):
    print('Read data, train:{0}, test:{1}'.format(train_path, test_path))
    config = data_reader.config
    train_data = data_reader.build_dataset(train_path)
    test_data = data_reader.build_dataset(test_path)

    with tf.Session() as sess:
        # Create model
        print('Create %d layers of %d units.' % (config.num_layers, config.size))
        model = create_model(sess, False, model_path, config=config)
        # Read data into buckets
        train_bucket_sizes = [len(train_data[b]) for b in range(len(config.buckets))]
        print("Training bucket sizes:{}".format(train_bucket_sizes))
        train_total_size = float(sum(train_bucket_sizes))
        print("Total train size:{}".format(train_total_size))

        # Bucket scale
        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < config.max_steps:
            # Choose a bucket according to data distribution. We pick a random
            # number in [0, 1] and use the corresponding interval in
            # train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_data, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / config.steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run
            # evals.
            if current_step % config.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float(
                    "inf")
                print("global step %d learning rate %.4f step-time %.2f "
                      "perplexity %.2f" % (
                          model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last
                #  3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(model_path, "translate.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(config.buckets)):
                    if len(test_data[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = \
                        model.get_batch(test_data, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs,
                                                 decoder_inputs,
                                                 target_weights, bucket_id,
                                                 True)
                    eval_ppx = math.exp(
                        float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (
                        bucket_id, eval_ppx))
                sys.stdout.flush()


def main(_):
    print('Training model...')
    data_reader = FCEReader(seq2seq_config, seq2seq_config.train_path)
    train(data_reader,
          seq2seq_config.train_path,
          seq2seq_config.val_path,
          seq2seq_config.model_path)


if __name__ == "__main__":
    tf.app.run()  # CPU, i5, about ten hours

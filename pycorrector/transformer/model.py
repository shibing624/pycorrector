# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

"""This example demonstrates how to train a standard Transformer model using
OpenNMT-tf as a library in about 200 lines of code. While relatively short,
this example contains some advanced concepts such as dataset bucketing and
prefetching, token-based batching, gradients accumulation, beam search, etc.
Currently, the beam search part is not easily customizable. This is expected to
be improved for TensorFlow 2.0 which is eager first.
"""

import argparse
import os

import opennmt as onmt
import tensorflow as tf
from opennmt import constants
from opennmt.utils import decay
from opennmt.utils import losses
from opennmt.utils import misc
from opennmt.utils import optim

# Define the "base" Transformer model.
source_inputter = onmt.inputters.WordEmbedder("source_vocabulary", embedding_size=512)
target_inputter = onmt.inputters.WordEmbedder("target_vocabulary", embedding_size=512)

encoder = onmt.encoders.SelfAttentionEncoder(
    num_layers=6,
    num_units=512,
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)
decoder = onmt.decoders.SelfAttentionDecoder(
    num_layers=6,
    num_units=512,
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)


def train(model_dir,
          example_inputter,
          source_file,
          target_file,
          maximum_length=100,
          shuffle_buffer_size=1000000,
          gradients_accum=8,
          train_steps=100000,
          save_every=1000,
          report_every=50):
    """Runs the training loop.
    Args:
      model_dir: Directory where checkpoints are saved.
      example_inputter: The inputter instance that produces the training examples.
      source_file: The source training file.
      target_file: The target training file.
      maximum_length: Filter sequences longer than this.
      shuffle_buffer_size: How many examples to load for shuffling.
      gradients_accum: Accumulate gradients of this many iterations.
      train_steps: Train for this many iterations.
      save_every: Save a checkpoint every this many iterations.
      report_every: Report training progress every this many iterations.
    """
    mode = tf.estimator.ModeKeys.TRAIN

    # Create the dataset.
    dataset = example_inputter.make_training_dataset(
        source_file,
        target_file,
        batch_size=3072,
        batch_type="tokens",
        shuffle_buffer_size=shuffle_buffer_size,
        bucket_width=1,  # Bucketize sequences by the same length for efficiency.
        maximum_features_length=maximum_length,
        maximum_labels_length=maximum_length)
    iterator = dataset.make_initializable_iterator()
    source, target = iterator.get_next()

    # Encode the source.
    with tf.variable_scope("encoder"):
        source_embedding = source_inputter.make_inputs(source, training=True)
        memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)

    # Decode the target.
    with tf.variable_scope("decoder"):
        target_embedding = target_inputter.make_inputs(target, training=True)
        logits, _, _ = decoder.decode(
            target_embedding,
            target["length"],
            vocab_size=target_inputter.vocabulary_size,
            mode=mode,
            memory=memory,
            memory_sequence_length=source["length"])

    # Compute the loss.
    loss, normalizer, _ = losses.cross_entropy_sequence_loss(
        logits,
        target["ids_out"],
        target["length"],
        label_smoothing=0.1,
        average_in_time=True,
        mode=mode)
    loss /= normalizer

    # Define the learning rate schedule.
    step = tf.train.create_global_step()
    learning_rate = decay.noam_decay_v2(2.0, step, model_dim=512, warmup_steps=4000)

    # Define the optimization op.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    train_op, optim_variables = optim.delayed_update(
        optimizer,
        gradients,
        step,
        accum_count=gradients_accum)

    # Runs the training loop.
    saver = tf.train.Saver()
    checkpoint_path = None
    if os.path.exists(model_dir):
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
    with tf.Session() as sess:
        if checkpoint_path is not None:
            print("Restoring parameters from %s" % checkpoint_path)
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(optim_variables))
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        last_step = -1
        while True:
            step_, lr_, loss_, _ = sess.run([step, learning_rate, loss, train_op])
            if step_ != last_step:
                if step_ % report_every == 0:
                    print("Step = %d ; Learning rate = %f ; Loss = %f" % (step_, lr_, loss_))
                if step_ % save_every == 0:
                    print("Saving checkpoint for step %d" % step_)
                    saver.save(sess, "%s/model" % model_dir, global_step=step_)
                if step_ == train_steps:
                    break
            last_step = step_


def translate(model_dir,
              example_inputter,
              source_file,
              batch_size=32,
              beam_size=4):
    """Runs translation.
    Args:
      model_dir: The directory to load the checkpoint from.
      example_inputter: The inputter instance that produces the training examples.
      source_file: The source file.
      batch_size: The batch size to use.
      beam_size: The beam size to use. Set to 1 for greedy search.
    """
    mode = tf.estimator.ModeKeys.PREDICT

    # Create the inference dataset.
    dataset = example_inputter.make_inference_dataset(source_file, batch_size)
    iterator = dataset.make_initializable_iterator()
    source = iterator.get_next()

    # Encode the source.
    with tf.variable_scope("encoder"):
        source_embedding = source_inputter.make_inputs(source)
        memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)

    # Generate the target.
    with tf.variable_scope("decoder"):
        target_inputter.build()
        batch_size = tf.shape(memory)[0]
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID
        target_ids, _, target_length, _ = decoder.dynamic_decode_and_search(
            target_inputter.embedding,
            start_tokens,
            end_token,
            vocab_size=target_inputter.vocabulary_size,
            beam_width=beam_size,
            memory=memory,
            memory_sequence_length=source["length"])
        target_vocab_rev = target_inputter.vocabulary_lookup_reverse()
        target_tokens = target_vocab_rev.lookup(tf.cast(target_ids, tf.int64))

    # Iterates on the dataset.
    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        while True:
            try:
                batch_tokens, batch_length = sess.run([target_tokens, target_length])
                for tokens, length in zip(batch_tokens, batch_length):
                    misc.print_bytes(b" ".join(tokens[0][:length[0] - 1]))
            except tf.errors.OutOfRangeError:
                break


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train", "translate"],
                        help="Run type.")
    parser.add_argument("--src", required=True,
                        help="Path to the source file.")
    parser.add_argument("--tgt",
                        help="Path to the target file.")
    parser.add_argument("--src_vocab", required=True,
                        help="Path to the source vocabulary.")
    parser.add_argument("--tgt_vocab", required=True,
                        help="Path to the target vocabulary.")
    parser.add_argument("--model_dir", default="checkpoint",
                        help="Directory where checkpoint are written.")
    args = parser.parse_args()

    inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": args.src_vocab,
        "target_vocabulary": args.tgt_vocab
    })

    if args.run == "train":
        train(args.model_dir, inputter, args.src, args.tgt)
    elif args.run == "translate":
        translate(args.model_dir, inputter, args.src)


if __name__ == "__main__":
    main()

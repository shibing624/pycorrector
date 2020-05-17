# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: This example demonstrates how to train a standard Transformer model using
OpenNMT-tf as a library in about 200 lines of code. While relatively short,
this example contains some advanced concepts such as dataset bucketing and
prefetching, token-based batching, gradients accumulation, beam search, etc.
requires: tensorflow==2.0.0, OpenNMT-tf==2.4.0
"""

import logging

import opennmt as onmt
import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel(logging.INFO)

# Define the model. For the purpose of this example, the model components
# (encoder, decoder, etc.) will be called separately.
model = onmt.models.SequenceToSequence(
    source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))

# Define the learning rate schedule and the optimizer.
learning_rate = onmt.schedules.NoamDecay(scale=2.0, model_dim=512, warmup_steps=8000)
optimizer = tfa.optimizers.LazyAdam(learning_rate)

# Track the model and optimizer weights.
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


def train(source_file,
          target_file,
          checkpoint_manager,
          batch_size=3072,
          maximum_length=100,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=100000,
          save_every=1000,
          report_every=100):
    """Runs the training loop.
    Args:
      source_file: The source training file.
      target_file: The target training file.
      checkpoint_manager: The checkpoint manager.
      batch_size: Batch size to train
      maximum_length: Filter sequences longer than this.
      shuffle_buffer_size: How many examples to load for shuffling.
      train_steps: Train for this many iterations.
      save_every: Save a checkpoint every this many iterations.
      report_every: Report training progress every this many iterations.
    """
    # Create the training dataset.
    dataset = model.examples_inputter.make_training_dataset(
        source_file,
        target_file,
        batch_size=batch_size,
        batch_type="tokens",
        shuffle_buffer_size=shuffle_buffer_size,
        length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
        maximum_features_length=maximum_length,
        maximum_labels_length=maximum_length)

    @tf.function(input_signature=dataset.element_spec)
    def training_step(source, target):
        # Run the encoder.
        source_inputs = model.features_inputter(source, training=True)
        encoder_outputs, _, _ = model.encoder(
            source_inputs,
            source["length"],
            training=True)

        # Run the decoder.
        target_inputs = model.labels_inputter(target, training=True)
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs,
            memory_sequence_length=source["length"])
        logits, _, _ = model.decoder(
            target_inputs,
            target["length"],
            state=decoder_state,
            training=True)

        # Compute the cross entropy loss.
        loss_num, loss_den, _ = onmt.utils.cross_entropy_sequence_loss(
            logits,
            target["ids_out"],
            target["length"],
            label_smoothing=0.1,
            average_in_time=True,
            training=True)
        loss = loss_num / loss_den

        # Compute and apply the gradients.
        variables = model.trainable_variables
        gradients = optimizer.get_gradients(loss, variables)
        optimizer.apply_gradients(list(zip(gradients, variables)))
        return loss

    # Runs the training loop.
    for source, target in dataset:
        loss = training_step(source, target)
        step = optimizer.iterations.numpy()
        if step % report_every == 0:
            tf.get_logger().info(
                "Step = %d ; Learning rate = %f ; Loss = %f",
                step, learning_rate(step), loss)
        if step % save_every == 0:
            tf.get_logger().info("Saving checkpoint for step %d", step)
            checkpoint_manager.save(checkpoint_number=step)
        if step == train_steps:
            break


def translate(source_file,
              batch_size=32,
              beam_size=4):
    """Runs translation. For error correction infer.
    Args:
      source_file: The source file.
      batch_size: The batch size to use.
      beam_size: The beam size to use. Set to 1 for greedy search.
    """
    # Create the inference dataset.
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)

    @tf.function(input_signature=(dataset.element_spec,))
    def predict(source):
        # Run the encoder.
        source_length = source["length"]
        batch_size = tf.shape(source_length)[0]
        source_inputs = model.features_inputter(source)
        encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

        # Prepare the decoding strategy.
        if beam_size > 1:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
            source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
            decoding_strategy = onmt.utils.BeamSearch(beam_size)
        else:
            decoding_strategy = onmt.utils.GreedySearch()

        # Run dynamic decoding.
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs,
            memory_sequence_length=source_length)
        decoded = model.decoder.dynamic_decode(
            model.labels_inputter,
            tf.fill([batch_size], onmt.START_OF_SENTENCE_ID),
            end_id=onmt.END_OF_SENTENCE_ID,
            initial_state=decoder_state,
            decoding_strategy=decoding_strategy,
            maximum_iterations=200)
        target_lengths = decoded.lengths
        target_tokens = model.labels_inputter.ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
        return target_tokens, target_lengths

    for source in dataset:
        batch_tokens, batch_length = predict(source)
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
            sentence = b" ".join(tokens[0][:length[0]])
            print(sentence.decode("utf-8"))

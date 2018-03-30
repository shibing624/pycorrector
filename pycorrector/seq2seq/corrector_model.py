from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import seq2seq
from reader import PAD_ID, GO_ID


class CorrectorModel(object):
    """Sequence-to-sequence model used to correct grammatical errors in text.

    NOTE: mostly copied from TensorFlow's seq2seq_model.py; only modifications
    are:
     - the introduction of RMSProp as an optional optimization algorithm
     - the introduction of a "projection bias" that biases decoding towards
       selecting tokens that appeared in the input
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False, config=None,
                 corrective_tokens_mask=None):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input
            length that will be processed in that bucket, and O specifies
            maximum output length. Training instances that have longer than I
            or outputs longer than O will be pushed to the next bucket and
            padded accordingly. We assume that the list is sorted, e.g., [(2,
            4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g.,
            for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when
            needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the
            model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.config = config

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(
                                                          i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(
                                                          i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(
                                                          i)))

        # One hot encoding of corrective tokens.
        corrective_tokens_tensor = tf.constant(corrective_tokens_mask if
                                               corrective_tokens_mask else
                                               np.zeros(self.target_vocab_size),
                                               shape=[self.target_vocab_size],
                                               dtype=tf.float32)
        batched_corrective_tokens = tf.stack(
            [corrective_tokens_tensor] * self.batch_size)
        self.batch_corrective_tokens_mask = batch_corrective_tokens_mask = \
            tf.placeholder(
                tf.float32,
                shape=[None, None],
                name="corrective_tokens")

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary
        # size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])

            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, labels, logits,
                                                  num_samples,
                                                  self.target_vocab_size)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            """

            :param encoder_inputs: list of length equal to the input bucket
            length of 1-D tensors (of length equal to the batch size) whose
            elements consist of the token index of each sample in the batch
            at a given index in the input.
            :param decoder_inputs:
            :param do_decode:
            :return:
            """

            if do_decode:
                # Modify bias here to bias the model towards selecting words
                # present in the input sentence.
                input_bias = self.build_input_bias(encoder_inputs,
                                                   batch_corrective_tokens_mask)

                # Redefined seq2seq to allow for the injection of a special
                # decoding function that
                return seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode,
                    loop_fn_factory=
                    apply_input_bias_and_extract_argmax_fn_factory(input_bias))
            else:
                return seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode)

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)

            if output_projection is not None:
                for b in range(len(buckets)):
                    # We need to apply the same input bias used during model
                    # evaluation when decoding.
                    input_bias = self.build_input_bias(
                        self.encoder_inputs[:buckets[b][0]],
                        batch_corrective_tokens_mask)
                    self.outputs[b] = [
                        project_and_apply_input_bias(output, output_projection,
                                                     input_bias)
                        for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.RMSPropOptimizer(0.001) if self.config.use_rms_prop \
                else tf.train.GradientDescentOptimizer(self.learning_rate)
            # opt = tf.train.AdamOptimizer()

            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def build_input_bias(self, encoder_inputs, batch_corrective_tokens_mask):
        packed_one_hot_inputs = tf.one_hot(indices=tf.stack(
            encoder_inputs, axis=1), depth=self.target_vocab_size)
        return tf.maximum(batch_corrective_tokens_mask,
                          tf.reduce_max(packed_one_hot_inputs,
                                        reduction_indices=1))

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only, corrective_tokens=None):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do
          backward), average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified
            bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights,
        # as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        corrective_tokens_vector = (corrective_tokens if
                                    corrective_tokens is not None else
                                    np.zeros(self.target_vocab_size))
        batch_corrective_tokens = np.repeat([corrective_tokens_vector],
                                            self.batch_size, axis=0)
        input_feed[self.batch_corrective_tokens_mask.name] = (
            batch_corrective_tokens)

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs.
            return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for
        step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for
        feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a
            batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...)
          later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (
                encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD
                # symbol. The corresponding target is decoder_input shifted by 1
                # forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


def project_and_apply_input_bias(logits, output_projection, input_bias):
    if output_projection is not None:
        logits = nn_ops.xw_plus_b(
            logits, output_projection[0], output_projection[1])

    # Apply softmax to ensure all tokens have a positive value.
    probs = tf.nn.softmax(logits)

    # Apply input bias, which is a mask of shape [batch, vocab len]
    # where each token from the input in addition to all "corrective"
    # tokens are set to 1.0.
    return tf.multiply(probs, input_bias)


def apply_input_bias_and_extract_argmax_fn_factory(input_bias):
    """

    :param encoder_inputs: list of length equal to the input bucket
    length of 1-D tensors (of length equal to the batch size) whose
    elements consist of the token index of each sample in the batch
    at a given index in the input.
    :return:
    """

    def fn_factory(embedding, output_projection=None, update_embedding=True):
        """Get a loop_function that extracts the previous symbol and embeds it.

        Args:
          embedding: embedding tensor for symbols.
          output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
          update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

        Returns:
          A loop function.
        """

        def loop_function(prev, _):
            prev = project_and_apply_input_bias(prev, output_projection,
                                                input_bias)

            prev_symbol = math_ops.argmax(prev, 1)
            # Note that gradients will not propagate through the second
            # parameter of embedding_lookup.
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = array_ops.stop_gradient(emb_prev)
            return emb_prev, prev_symbol

        return loop_function

    return fn_factory

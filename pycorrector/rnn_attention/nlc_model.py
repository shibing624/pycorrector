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

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell, rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh

from pycorrector.rnn_attention.nlc_data import SOS_ID, EOS_ID


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tanh(rnn_cell_impl._linear(hs2d, num_units, False))

                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell_impl._linear(gru_out, self._num_units, False))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
            weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell_impl._linear([context, gru_out], self._num_units, False))
            self.attn_map = tf.squeeze(tf.slice(weights, [0, 0, 0], [-1, -1, 1]))
            return (out, out)


class NLCModel(object):
    def __init__(self, vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, dropout, forward_only=False, optimizer="adam"):

        self.size = size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.source_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.target_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.target_length = tf.reduce_sum(self.target_mask, reduction_indices=0)

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

        with tf.variable_scope("NLC", initializer=tf.initializers.variance_scaling(1.0)):
            self.setup_embeddings()
            self.setup_encoder()
            self.setup_decoder()
            self.setup_loss()

            self.setup_beam()

        params = tf.trainable_variables()
        if not forward_only:
            opt = get_optimizer(optimizer)(self.learning_rate)

            gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            #      self.gradient_norm = tf.global_norm(clipped_gradients)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [self.vocab_size, self.size])
            self.L_dec = tf.get_variable("L_dec", [self.vocab_size, self.size])
            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.source_tokens)
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)

    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size)
        with vs.variable_scope("PryamidEncoder"):
            inp = self.encoder_inputs
            mask = self.source_mask
            out = None
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    srclen = tf.reduce_sum(mask, reduction_indices=0)
                    out, _ = self.bidirectional_rnn(self.encoder_cell, inp, srclen, scope=scope)
                    dropin, mask = self.downscale(out, mask)
                    inp = self.dropout(dropin)
            self.encoder_output = out

    def setup_decoder(self):
        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size)
        self.attn_cell = GRUCellAttn(self.size, self.encoder_output, scope="DecoderAttnCell")

        with vs.variable_scope("Decoder"):
            inp = self.decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, inp, time_major=True,
                                                        dtype=dtypes.float32, sequence_length=self.target_length,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = self.dropout(out)
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, inp, time_major=True,
                                                    dtype=dtypes.float32, sequence_length=self.target_length,
                                                    scope=scope, initial_state=self.decoder_state_input[i + 1])
                self.decoder_output = self.dropout(out)
                self.decoder_state_output.append(state_output)

    def decoder_graph(self, decoder_inputs, decoder_state_input):
        decoder_output, decoder_state_output = None, []
        inp = decoder_inputs

        with vs.variable_scope("Decoder", reuse=True):
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    inp, state_output = self.decoder_cell(inp, decoder_state_input[i])
                    decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                decoder_output, state_output = self.attn_cell(inp, decoder_state_input[i + 1])
                decoder_state_output.append(state_output)

        return decoder_output, decoder_state_output

    def setup_beam(self):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[SOS_ID]])
        beam_probs_0 = tf.constant([0.])

        cand_seqs_0 = tf.constant([[EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])

        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        def beam_cond(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            return tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs)

        def beam_step(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = embedding_ops.embedding_lookup(self.L_dec, inputs)
            decoder_output, state_output = self.decoder_graph(decoder_input, states)

            with vs.variable_scope("Logistic", reuse=True):
                do2d = tf.reshape(decoder_output, [-1, self.size])
                logits2d = rnn_cell_impl._linear(do2d, self.vocab_size, False)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [batch_size, EOS_ID]),
                                           tf.tile([[-3e38]], [batch_size, 1]),
                                           tf.slice(total_probs, [0, EOS_ID + 1],
                                                    [batch_size, self.vocab_size - EOS_ID - 1])], 1)

            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.vocab_size)
            next_mods = tf.mod(top_indices, self.vocab_size)

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                                        tf.reshape(next_mods, [-1, 1])], 1)

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            EOS_probs = tf.slice(total_probs, [0, EOS_ID], [batch_size, 1])
            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], 0)

            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            return [time + 1, next_beam_probs, next_beam_seqs, next_cand_probs, next_cand_seqs] + next_states

        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((cand_probs_0, tf.TensorShape([None, ])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes,
                                 back_prop=False)
        #    time, beam_probs, beam_seqs, cand_probs, cand_seqs, _ = ret_vars
        cand_seqs = ret_vars[4]
        cand_probs = ret_vars[3]
        self.beam_output = cand_seqs
        self.beam_scores = cand_probs

    def setup_loss(self):
        with vs.variable_scope("Logistic"):
            doshape = tf.shape(self.decoder_output)
            T, batch_size = doshape[0], doshape[1]
            do2d = tf.reshape(self.decoder_output, [-1, self.size])
            logits2d = rnn_cell_impl._linear(do2d, self.vocab_size, False)
            outputs2d = tf.nn.log_softmax(logits2d)
            self.outputs = tf.reshape(outputs2d, tf.stack([T, batch_size, self.vocab_size]))

            targets_no_GO = tf.slice(self.target_tokens, [1, 0], [-1, -1])
            masks_no_GO = tf.slice(self.target_mask, [1, 0], [-1, -1])
            # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
            mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])
            losses1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2d, labels=labels1d) * tf.to_float(
                mask1d)
            losses2d = tf.reshape(losses1d, tf.stack([T, batch_size]))
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def downscale(self, inp, mask):
        # return inp, mask

        with vs.variable_scope("Downscale"):
            inshape = tf.shape(inp)
            T, batch_size, dim = inshape[0], inshape[1], inshape[2]
            inp2d = tf.reshape(tf.transpose(inp, perm=[1, 0, 2]), [-1, 2 * self.size])
            out2d = rnn_cell_impl._linear(inp2d, self.size, False)
            out3d = tf.reshape(out2d, tf.stack((batch_size, tf.to_int32(T / 2), dim)))
            out3d = tf.transpose(out3d, perm=[1, 0, 2])
            out3d.set_shape([None, None, self.size])
            out = tanh(out3d)

            mask = tf.transpose(mask)
            mask = tf.reshape(mask, [-1, 2])
            mask = tf.cast(mask, tf.bool)
            mask = tf.reduce_any(mask, reduction_indices=1)
            mask = tf.to_int32(mask)
            mask = tf.reshape(mask, tf.stack([batch_size, -1]))
            mask = tf.transpose(mask)
        return out, mask

    def bidirectional_rnn(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with vs.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=0, batch_dim=1)
        with vs.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=0, batch_dim=1)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw

        return (outputs, output_state)

    def set_default_decoder_state_input(self, input_feed, batch_size):
        default_value = np.zeros([batch_size, self.size])
        for i in xrange(self.num_layers):
            input_feed[self.decoder_state_input[i]] = default_value

    def train(self, session, source_tokens, source_mask, target_tokens, target_mask):
        input_feed = {self.source_tokens: source_tokens, self.target_tokens: target_tokens,
                      self.source_mask: source_mask, self.target_mask: target_mask,
                      self.keep_prob: self.keep_prob_config}
        self.set_default_decoder_state_input(input_feed, batch_size=target_tokens.shape[1])

        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3]

    def test(self, session, source_tokens, source_mask, target_tokens, target_mask):
        input_feed = {self.source_tokens: source_tokens, self.target_tokens: target_tokens,
                      self.source_mask: source_mask, self.target_mask: target_mask,
                      self.keep_prob: 1.}
        self.set_default_decoder_state_input(input_feed, batch_size=target_tokens.shape[1])

        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def encode(self, session, source_tokens, source_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.keep_prob] = 1.

        output_feed = [self.encoder_output]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def decode(self, session, encoder_output, target_tokens, target_mask=None, decoder_states=None):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.target_tokens] = target_tokens
        input_feed[self.target_mask] = target_mask if target_mask else np.ones_like(target_tokens)
        input_feed[self.keep_prob] = 1.

        if not decoder_states:
            self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])
        else:
            for i in xrange(self.num_layers):
                input_feed[self.decoder_state_input[i]] = decoder_states[i]

        output_feed = [self.outputs] + self.decoder_state_output
        outputs = session.run(output_feed, input_feed)
        return outputs[0], None, outputs[1:]

    def decode_beam(self, session, encoder_output, beam_size=8):
        input_feed = {self.encoder_output: encoder_output, self.keep_prob: 1., self.beam_size: beam_size}

        output_feed = [self.beam_output, self.beam_scores]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

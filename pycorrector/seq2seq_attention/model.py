# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import time

import numpy as np
import tensorflow as tf

from .data_reader import GO_TOKEN, EOS_TOKEN, preprocess_sentence


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc(output)
        return x, state, attention_weights


class Seq2SeqModel(object):
    def __init__(self, source_word2id, target_word2id, embedding_dim=256, hidden_dim=1024,
                 batch_size=64, maxlen=128, checkpoint_path='', gpu_id=0):
        if gpu_id > -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.vocab_src_size = len(source_word2id) + 1
        self.vocab_tar_size = len(target_word2id) + 1
        self.encoder = Encoder(self.vocab_src_size, embedding_dim, hidden_dim, batch_size)
        self.attention_layer = BahdanauAttention(10)
        self.decoder = Decoder(self.vocab_tar_size, embedding_dim, hidden_dim, batch_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.source_word2id = source_word2id
        self.target_word2id = target_word2id
        self.maxlen = maxlen
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        # Load model
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print('last checkpoit restore, checkpoint path:', checkpoint_path)

    def train(self, example_source_batch, dataset, steps_per_epoch, epochs=10):
        # sample input
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder(example_source_batch, sample_hidden)
        print('Encoder output shape: (batch size, sequence length, hidden_dim) {}'.format(sample_output.shape))
        print('Encoder Hidden state shape: (batch size, hidden_dim) {}'.format(sample_hidden.shape))

        attention_result, attention_weights = self.attention_layer(sample_hidden, sample_output)
        print("Attention result shape: (batch size, hidden_dim) {}".format(attention_result.shape))
        print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

        sample_decoder_output, _, _ = self.decoder(tf.random.uniform((self.batch_size, 1)),
                                                   sample_hidden, sample_output)
        print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        @tf.function
        def train_step(inp, targ, enc_hidden):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(inp, enc_hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([self.target_word2id[GO_TOKEN]] * self.batch_size, 1)
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            # 求梯度
            gradients = tape.gradient(loss, variables)
            # 反向传播
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        print("start train...")
        for epoch in range(epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('epoch {}, save model at {}'.format(
                    epoch + 1, ckpt_save_path
                ))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def evaluate(self, sentence):
        attention_plot = np.zeros((self.maxlen, self.maxlen))
        char_split_sent = ' '.join(list(sentence.replace(" ", "")))
        sentence = preprocess_sentence(char_split_sent)

        inputs = [self.source_word2id[i] for i in sentence.split(' ') if i in self.source_word2id]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.maxlen,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, self.hidden_dim))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.target_word2id[GO_TOKEN]], 0)
        target_id2word = {v: k for k, v in self.target_word2id.items()}
        for t in range(self.maxlen):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)
            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += target_id2word[predicted_id] + ' '
            if target_id2word[predicted_id] == EOS_TOKEN:
                return result, sentence, attention_plot
            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot

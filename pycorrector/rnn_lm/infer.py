# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import numpy as np
import tensorflow as tf

import rnn_lm_config as conf
from rnn_lm.data_reader import process_data
from rnn_lm_model import rnn_model


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def generate(begin_word):
    batch_size = 1
    print('loading corpus from %s' % conf.model_dir)
    data_vector, word_idx, vocabularies = process_data(conf.train_word_path)
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(model='lstm',
                           input_data=input_data,
                           output_data=None,
                           vocab_size=len(vocabularies),
                           rnn_size=128,
                           num_layers=2,
                           batch_size=64,
                           learning_rate=0.0002)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(conf.model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_idx.get, conf.start_token))])
        [predict, last_state] = sess.run([end_points['prediction'],
                                          end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        sentence = ''

        i = 0
        while word != conf.end_token:
            sentence += word
            i += 1
            if i >= 24:
                break
            x = np.zeros((1, 1))
            try:
                x[0, 0] = word_idx[word]
            except KeyError:
                print("please enter a chinese char again.")
                break
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return sentence


def ppl(text):
    perplexity = 0
    batch_size = 1
    print('loading corpus from %s' % conf.model_dir)
    data_vector, word_idx, vocabularies = process_data(conf.train_word_path)
    x = [word_idx[c] if c in word_idx else word_idx[' '] for c in text]
    x = [word_idx[conf.start_token]] + x + [word_idx[conf.end_token]]
    # pad x so the batch_size divides it
    while len(x) % conf.batch_size != 1:
        x.append(word_idx[' '])
    y = np.array(x[1:]).reshape((-1, batch_size))
    x = np.array(x[:-1]).reshape((-1, batch_size))

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(model='lstm',
                           input_data=input_data,
                           output_data=output_targets,
                           vocab_size=len(vocabularies),
                           rnn_size=128,
                           num_layers=2,
                           batch_size=batch_size,
                           learning_rate=conf.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(conf.model_dir)
        saver.restore(sess, checkpoint)
        for i in range(x.shape[0]):
            [total_loss, last_state, perplexity] = sess.run([end_points['total_loss'],
                                                             end_points['last_state'],
                                                             end_points['perplexity']],
                                                            feed_dict={input_data: x[i:i + 1, :],
                                                                       output_targets: y[i:i + 1, :]})
        print('perplexity: {0}'.format(perplexity))
    return perplexity


if __name__ == '__main__':
    # begin_char = input('please input the first character:')
    # print(generate(begin_char))
    ppl("化肥和农药不仅对人类有害。")
    # ppl("化肥和农药不仅对人、类这最化有害。")

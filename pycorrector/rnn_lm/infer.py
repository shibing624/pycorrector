# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import numpy as np
import tensorflow as tf

import rnn_lm_config as conf
from rnn_lm.data_reader import UNK_TOKEN, END_TOKEN, START_TOKEN, load_word_dict
from rnn_lm_model import rnn_model


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = 0
    return vocabs[sample]


def generate(begin_word):
    batch_size = 1
    word_to_int = load_word_dict(conf.word_dict_path)
    vocabularies = [k for k, v in word_to_int.items()]
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(model='lstm',
                           input_data=input_data,
                           output_data=None,
                           vocab_size=len(word_to_int),
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
        print("loading model from the checkpoint {0}".format(checkpoint))
        x = np.array([list(map(word_to_int.get, START_TOKEN))])
        [predict, last_state] = sess.run([end_points['prediction'],
                                          end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        sentence = ''
        i = 0
        while word != END_TOKEN and word != START_TOKEN and word!=UNK_TOKEN:
            sentence += word
            i += 1
            if i >= 24:
                break
            x = np.zeros((1, 1))
            try:
                x[0, 0] = word_to_int[word]
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
    word_to_int = load_word_dict(conf.word_dict_path)
    # data idx
    x = [word_to_int[c] if c in word_to_int else word_to_int[UNK_TOKEN] for c in text]
    x = [word_to_int[START_TOKEN]] + x + [word_to_int[END_TOKEN]]
    # reshape
    y = np.array(x[1:]).reshape((-1, batch_size))
    x = np.array(x[:-1]).reshape((-1, batch_size))

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(model='lstm',
                           input_data=input_data,
                           output_data=output_targets,
                           vocab_size=len(word_to_int),
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
        print("loading model from the checkpoint {0}".format(checkpoint))
        for i in range(x.shape[0]):
            [perplexity] = sess.run([end_points['perplexity']],
                                    feed_dict={input_data: x[i:i + 1, :],
                                               output_targets: y[i:i + 1, :]})
        print('perplexity: {0}'.format(perplexity))
    return perplexity


if __name__ == '__main__':
    # begin_char = input('please input the first character:')
    begin_char = '我'
    print(generate(begin_char))
    # ppl("化肥和农药不仅对人类有害，而且对地球的土壤也非常有害。")
    # ppl("化肥和农药不仅对土壤也非常。")
    # ppl("化肥和农药、不仅对人、类这最化有害。")

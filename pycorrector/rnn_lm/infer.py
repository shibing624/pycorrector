# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import tensorflow as tf
from rnn_lm_model import rnn_model
from rnn_lm.data_reader import process_data
import numpy as np
import rnn_lm_config as conf

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
        while word != end_token:
            sentence += word
            i += 1
            if i >= 24:
                break
            x = np.zeros((1, 1))
            x[0, 0] = word_idx[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return sentence


if __name__ == '__main__':
    begin_char = input('please input the first character:')
    data = generate(begin_char)
    print(data)
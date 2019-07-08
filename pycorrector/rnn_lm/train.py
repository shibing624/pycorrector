# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: train rnn language model
import os
import sys

import tensorflow as tf

sys.path.append('../..')
from pycorrector.rnn_lm import config
from pycorrector.rnn_lm.data_reader import process_data, generate_batch
from pycorrector.rnn_lm.rnn_lm_model import rnn_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(_):
    # build vocab and word dict
    data_vector, word_to_int = process_data(config.train_word_path, config.word_dict_path, config.cutoff_frequency)
    # batch data
    batches_inputs, batches_outputs = generate_batch(config.batch_size, data_vector, word_to_int)
    # placeholder
    input_data = tf.placeholder(tf.int32, [config.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [config.batch_size, None])
    # create model
    end_points = rnn_model(model='lstm',
                           input_data=input_data,
                           output_data=output_targets,
                           vocab_size=len(word_to_int),
                           rnn_size=128,
                           num_layers=2,
                           batch_size=config.batch_size,
                           learning_rate=config.learning_rate)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # start
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        # init
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(config.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('start training...')
        try:
            for epoch in range(start_epoch, config.epochs):
                n = 0
                n_chunk = len(data_vector) // config.batch_size
                for batch in range(n_chunk):
                    loss, _, _, perplexity = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op'],
                        end_points['perplexity']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f, ppl: %.1f' % (epoch, batch, loss, perplexity))
                if epoch % config.num_save_epochs == 0:
                    saver.save(sess, os.path.join(config.model_dir, config.model_prefix), global_step=epoch)
                    print('save model to %s,  epoch:%d' % (config.model_dir + config.model_prefix, epoch))
        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(config.model_dir, config.model_prefix), global_step=epoch)
            print('Last epoch were saved, next time will start from epoch {}.'.format(epoch))


if __name__ == '__main__':
    tf.app.run()

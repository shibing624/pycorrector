# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

sys.path.append('../..')

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.model import Seq2SeqModel

from pycorrector.seq2seq_attention.data_reader import create_dataset
from pycorrector.seq2seq_attention.train import tokenize


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence, attn_img_path=''):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    if attn_img_path:
        plt.savefig(attn_img_path)
    plt.clf()


def infer(model, sentence='由我起开始做。'):
    result, sentence, attention_plot = model.evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '), config.attention_image_path)


if __name__ == "__main__":
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    source_texts, target_texts = create_dataset(config.train_path, None)
    source_seq, source_word2id = tokenize(source_texts, config.maxlen)
    target_seq, target_word2id = tokenize(target_texts, config.maxlen)
    dataset = tf.data.Dataset.from_tensor_slices((source_seq, target_seq)).shuffle(len(source_seq))
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    example_source_batch, example_target_batch = next(iter(dataset))
    model = Seq2SeqModel(example_source_batch, source_word2id, target_word2id, embedding_dim=config.embedding_dim,
                         hidden_dim=config.hidden_dim,
                         batch_size=config.batch_size, maxlen=config.maxlen, checkpoint_path=config.model_dir,
                         gpu_id=config.gpu_id)
    for i in inputs:
        infer(model, i)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

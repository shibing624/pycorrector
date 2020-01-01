# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('../..')

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.model import Seq2SeqModel
from pycorrector.seq2seq_attention.data_reader import load_word_dict


def plot_attention(attention, sentence, predicted_sentence, attn_img_path=''):
    """
    Plotting the attention weights
    :param attention:
    :param sentence:
    :param predicted_sentence:
    :param attn_img_path:
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import font_manager
    my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 12}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, fontproperties=my_font)  # rotation=90,
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict, fontproperties=my_font)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    if attn_img_path:
        plt.savefig(attn_img_path)
    plt.clf()


def infer(model, sentence, attention_image_path=''):
    result, sentence, attention_plot = model.evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    if attention_image_path:
        try:
            plot_attention(attention_plot, sentence.split(' '), result.split(' '), attention_image_path)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    source_word2id = load_word_dict(config.save_src_vocab_path)
    target_word2id = load_word_dict(config.save_trg_vocab_path)
    model = Seq2SeqModel(source_word2id, target_word2id, embedding_dim=config.embedding_dim,
                         hidden_dim=config.hidden_dim,
                         batch_size=config.batch_size, maxlen=config.maxlen, checkpoint_path=config.model_dir,
                         gpu_id=config.gpu_id)
    for id, i in enumerate(inputs):
        img_path = os.path.join(config.output_dir, str(id) + ".png")
        infer(model, i, img_path)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

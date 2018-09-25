# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import json
import os
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Input, Lambda, Layer, Embedding, CuDNNLSTM, Bidirectional, Dense, Activation, GRU
from keras.models import Model
from keras.optimizers import Adam

from pycorrector.seq2seq import cged_config as config
from pycorrector.seq2seq.corpus_reader import CGEDReader, EOS_TOKEN, GO_TOKEN, PAD_TOKEN

maxlen = 400
batch_size = 64
epochs = 20
char_size = 128

train_path = config.train_path
vocab_json_path = config.vocab_json_path
data_reader = CGEDReader(train_path)
input_texts, target_texts = data_reader.build_dataset(train_path)

if os.path.exists(vocab_json_path):
    chars, id2char, char2id = json.load(open(vocab_json_path))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    chars = {}
    print('Training data...')
    print('input_texts:', input_texts[0])
    print('target_texts:', target_texts[0])
    max_input_texts_len = max([len(text) for text in input_texts])

    print('num of samples:', len(input_texts))
    print('max sequence length for inputs:', max_input_texts_len)

    chars = data_reader.read_vocab(input_texts + target_texts)
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([chars, id2char, char2id], open(vocab_json_path, 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [char2id[PAD_TOKEN]] * (ml - len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y = [], []
    while True:
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i]))
            Y.append(str2id(target_texts[i], start_end=False))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


# 搭建seq2seq模型

x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x = x_in
y = y_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)


def to_one_hot(x):  # 输出一个词表大小的向量，来标记该词是否在文章出现过
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars))
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars), char_size)
x = embedding(x)
y = embedding(y)

# encoder，双层双向LSTM
x = Bidirectional(GRU(int(char_size / 2), return_sequences=True))(x)
x = Bidirectional(GRU(int(char_size / 2), return_sequences=True))(x)

# decoder，双层单向LSTM
y = GRU(char_size, return_sequences=True)(y)
y = GRU(char_size, return_sequences=True)(y)


class Interact(Layer):
    """交互层，负责融合encoder和decoder的信息
    """

    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True)  # maxpooling1d
        mv = mv + K.zeros_like(q[:, :, :1])  # 将mv重复至“q的timesteps”份
        # 下面几步只是实现了一个乘性attention
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        # 将各步结果拼接
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2] + input_shape[1][2] * 2)


xy = Interact()([y, x, x_mask])
xy = Dense(512, activation='relu')(xy)
xy = Dense(len(chars))(xy)
xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
xy = Activation('softmax')(xy)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in, y_in], xy)
model.add_loss(loss)
model.compile(optimizer=Adam(1e-3))


def gen_target(s, topk=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk)  # 输入转id
    yid = np.array([[char2id[GO_TOKEN]]] * topk)  # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk  # 候选答案分数
    for i in range(50):  # 强制要求标题不超过50字
        proba = model.predict([xid, yid])[:, i, :]  # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _yid = []  # 暂存的候选目标序列
        _scores = []  # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk):  # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == char2id[EOS_TOKEN]:  # 找到<end>就返回
                return id2str(_yid[k][1:-1])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)][1:-1])


s1 = '吸烟的行为女人比男人更重 ，'
s2 = '所以在这期间，'


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print('input:' + s1)
        print('output:' + gen_target(s1))
        print('input:' + s2)
        print('output:' + gen_target(s2))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(config.attn_model_path)


model.load_weights(config.attn_model_path)

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
for i in inputs:
    target = gen_target(i)
    print('input:' + i)
    print('output:' + target)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

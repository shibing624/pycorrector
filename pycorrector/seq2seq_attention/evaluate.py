# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import numpy as np
from keras.callbacks import Callback,EarlyStopping

from pycorrector.seq2seq_attention.corpus_reader import str2id, id2str
from pycorrector.seq2seq_attention.reader import GO_TOKEN, EOS_TOKEN


def gen_target(input_text, model, char2id, id2char, maxlen=400, topk=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(input_text, char2id, maxlen)] * topk)  # 输入转id
    yid = np.array([[char2id[GO_TOKEN]]] * topk)  # 解码均以GO开始
    scores = [0] * topk  # 候选答案分数
    for i in range(50):  # 强制要求target不超过50字
        proba = model.predict([xid, yid])[:, i, :]  # 预测
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
                return id2str(_yid[k][1:-1], id2char)
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到EOS，直接返回
    return id2str(yid[np.argmax(scores)][1:-1], id2char)


class Evaluate(Callback):
    def __init__(self, model, attn_model_path, char2id, id2char, maxlen):
        super(Evaluate, self).__init__()
        self.lowest = 1e10
        self.model = model
        self.attn_model_path = attn_model_path
        self.char2id = char2id
        self.id2char = id2char
        self.maxlen = maxlen

    def on_epoch_end(self, epoch, logs=None):
        sents = ['吸烟的行为经常会影响社会里所有的人。',
                 '所以在这期间，']
        # 训练过程中观察一两个例子，显示预测质量提高的过程
        for sent in sents:
            target = gen_target(sent, self.model, self.char2id, self.id2char, self.maxlen)
            print('input:' + sent)
            print('output:' + target)
        # 保存最优结果
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            self.model.save_weights(self.attn_model_path)

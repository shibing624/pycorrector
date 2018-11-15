# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os
import sys

sys.path.append('../..')
import time
from codecs import open

import numpy as np
import torch
from torch import optim

from pycorrector.deep_context import config
from pycorrector.deep_context.data_util import write_embedding, write_config
from pycorrector.deep_context.network import Context2vec
from pycorrector.deep_context.reader import Dataset


def train(train_path: str,
          emb_path: str,
          model_path: str,
          use_mlp=True,
          batch_size=64,
          epochs=3,
          maxlen=64,
          word_embed_size=200,
          hidden_size=200,
          learning_rate=0.0001,
          n_layers=1,
          min_freq=1,
          dropout=0.0,
          gpu_id=0):
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    if not os.path.isfile(train_path):
        raise FileNotFoundError

    print('Loading input file')
    counter = 0
    with open(train_path, mode='r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().lower().split()
            if 0 < len(sentence) < maxlen:
                counter += 1

    sentences = np.empty(counter, dtype=object)
    counter = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().lower().split()
            if 0 < len(sentence) < maxlen:
                sentences[counter] = np.array(sentence)
                counter += 1

    print('Creating dataset')
    dataset = Dataset(sentences, batch_size, min_freq, device)
    counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                        for word in dataset.vocab.itos])
    model = Context2vec(vocab_size=len(dataset.vocab),
                        counter=counter,
                        word_embed_size=word_embed_size,
                        hidden_size=hidden_size,
                        n_layers=n_layers,
                        bidirectional=True,
                        use_mlp=use_mlp,
                        dropout=dropout,
                        pad_index=dataset.pad_index,
                        device=device,
                        inference=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('batch_size:', batch_size, 'epochs:', epochs, 'word_embed_size:', word_embed_size, 'hidden_size:',
          hidden_size, 'device:', device)
    print('model:', model)

    interval = 1e6
    for epoch in range(epochs):
        begin_time = time.time()
        cur_at = begin_time
        total_loss = 0.0
        word_count = 0
        next_count = interval
        last_accum_loss = 0.0
        last_word_count = 0
        for iterator in dataset.get_batch_iter(batch_size):
            for batch in iterator:
                sentence = getattr(batch, 'sentence')
                target = sentence[:, 1:-1]
                if target.size(0) == 0:
                    continue
                optimizer.zero_grad()
                loss = model(sentence, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.data.mean()

                minibatch_size, sentence_length = target.size()
                word_count += minibatch_size * sentence_length
                accum_mean_loss = float(total_loss) / word_count if total_loss > 0.0 else 0.0
                if word_count >= next_count:
                    now = time.time()
                    duration = now - cur_at
                    throuput = float((word_count - last_word_count)) / (now - cur_at)
                    cur_mean_loss = (float(total_loss) - last_accum_loss) / (word_count - last_word_count)
                    print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'
                          .format(word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                    next_count += interval
                    cur_at = now
                    last_accum_loss = float(total_loss)
                    last_word_count = word_count

        print('total_loss:', total_loss.item())

    output_dir = os.path.dirname(emb_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, emb_path)
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), model_path + '.optim')
    output_config_file = model_path + '.config.json'
    write_config(output_config_file,
                 vocab_size=len(dataset.vocab),
                 word_embed_size=word_embed_size,
                 hidden_size=hidden_size,
                 n_layers=n_layers,
                 bidirectional=True,
                 use_mlp=use_mlp,
                 dropout=dropout,
                 pad_index=dataset.pad_index,
                 unk_token=dataset.unk_token,
                 bos_token=dataset.bos_token,
                 eos_token=dataset.eos_token,
                 learning_rate=learning_rate)


if __name__ == "__main__":
    train(config.train_path,
          config.emb_path,
          config.model_path,
          config.use_mlp,
          config.batch_size,
          config.epochs,
          config.maxlen,
          config.word_embed_size,
          config.hidden_size,
          config.learning_rate,
          config.n_layers,
          config.min_freq,
          config.dropout,
          config.gpu_id)

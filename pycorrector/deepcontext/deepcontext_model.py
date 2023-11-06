# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: context to vector network
"""

import os
import time
from typing import List

import numpy as np
import torch
from loguru import logger
from torch import optim

from pycorrector.deepcontext.deepcontext_utils import (
    Context2vec,
    read_config,
    load_word_dict,
    write_config,
    ContextDataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepContextModel:
    def __init__(self, model_dir: str, max_length: int = 512):
        # device
        logger.debug("Device: {}".format(device))
        self.config_file = os.path.join(model_dir, 'config.json')
        self.checkpoint_file = os.path.join(model_dir, "pytorch_model.bin")
        self.optimizer_file = os.path.join(model_dir, 'optimizer.pt')
        self.vocab_file = os.path.join(model_dir, 'vocab.txt')
        self.model_dir = model_dir
        self.max_length = max_length
        self.mask = "[]"
        self.model = None
        self.optimizer = None
        self.config_dict = None
        self.stoi = None
        self.itos = None

    def load_model(self):
        if not os.path.exists(self.config_file):
            raise ValueError('config file not exists.')
        if not os.path.exists(self.checkpoint_file):
            raise ValueError('checkpoint file not exists.')
        if not os.path.exists(self.vocab_file):
            raise ValueError('vocab file not exists.')
        config_dict = read_config(self.config_file)
        self.model = Context2vec(
            vocab_size=config_dict['vocab_size'],
            counter=[1] * config_dict['vocab_size'],
            word_embed_size=config_dict['word_embed_size'],
            hidden_size=config_dict['hidden_size'],
            n_layers=config_dict['n_layers'],
            use_mlp=config_dict['use_mlp'],
            dropout=config_dict['dropout'],
            pad_index=config_dict['pad_index'],
            device=device,
            is_inference=True
        ).to(device)
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer = optim.Adam(self.model.parameters(), lr=config_dict['learning_rate'])
        if os.path.exists(self.optimizer_file):
            self.optimizer.load_state_dict(torch.load(self.optimizer_file))
        self.config_dict = config_dict
        # read vocab
        self.stoi = load_word_dict(self.vocab_file)
        self.itos = {v: k for k, v in self.stoi.items()}

    def train_model(
            self,
            train_path,
            batch_size=64,
            num_epochs=3,
            word_embed_size=200,
            hidden_size=200,
            learning_rate=1e-3,
            n_layers=2,
            min_freq=1,
            dropout=0.0
    ):
        if not os.path.isfile(train_path):
            raise FileNotFoundError

        logger.info('Loading data')
        dataset = ContextDataset(
            train_path,
            batch_size,
            min_freq,
            device,
            self.vocab_file,
            self.max_length,
        )
        counter = np.array([dataset.word_freqs[word] for word in dataset.vocab_2_ids])
        model = Context2vec(
            vocab_size=len(dataset.vocab_2_ids),
            counter=counter,
            word_embed_size=word_embed_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_mlp=True,
            dropout=dropout,
            pad_index=dataset.pad_index,
            device=device,
            is_inference=False
        ).to(device)
        if self.model is None:
            # norm weight
            model.norm_embedding_weight(model.criterion.W)
        if self.optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = self.optimizer
        logger.info(
            'model: {model}, batch_size: {batch_size}, epochs: {epochs}, '
            'word_embed_size: {word_embed_size}, hidden_size: {hidden_size}, learning_rate: {learning_rate}'
        )

        # save config
        write_config(
            self.config_file,
            vocab_size=len(dataset.vocab_2_ids),
            word_embed_size=word_embed_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            use_mlp=True,
            dropout=dropout,
            pad_index=dataset.pad_index,
            pad_token=dataset.pad_token,
            unk_token=dataset.unk_token,
            sos_token=dataset.sos_token,
            eos_token=dataset.eos_token,
            learning_rate=learning_rate
        )

        interval = 1e5
        best_loss = 1e3
        logger.info("train start...")
        for epoch in range(num_epochs):
            begin_time = time.time()
            cur_at = begin_time
            total_loss = 0.0
            word_count = 0
            next_count = interval
            last_accum_loss = 0.0
            last_word_count = 0
            cur_loss = 0
            for it, (mb_x, mb_x_len) in enumerate(dataset.train_data):
                sentence = torch.from_numpy(mb_x).to(device).long()

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
                cur_mean_loss = (float(total_loss) - last_accum_loss) / (word_count - last_word_count)
                cur_loss = cur_mean_loss
                if word_count >= next_count:
                    now = time.time()
                    duration = now - cur_at
                    throuput = float((word_count - last_word_count)) / (now - cur_at)
                    logger.info('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'
                                .format(word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                    next_count += interval
                    cur_at = now
                    last_accum_loss = float(total_loss)
                    last_word_count = word_count

            # find best model
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)
            logger.info('epoch:[{}/{}], total_loss:[{}], best_cur_loss:[{}]'
                        .format(epoch + 1, num_epochs, total_loss.item(), best_loss))
            if is_best:
                self.save_model(model_dir=self.model_dir, model=model, optimizer=optimizer)
                logger.info('epoch:{}, save new model:{}'.format(epoch + 1, self.model_dir))

    def save_model(self, model_dir=None, model=None, optimizer=None):
        """Save the model and the optim."""
        if not model_dir:
            model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)

        if model:
            # Take care of distributed/parallel training
            torch.save(model.state_dict(), self.checkpoint_file)
        if optimizer:
            torch.save(optimizer.state_dict(), self.optimizer_file)

    def predict_mask_token(self, tokens: List[str], mask_index: int = 0, k: int = 10):
        if not self.model:
            self.load_model()
        unk_token = self.config_dict['unk_token']
        sos_token = self.config_dict['sos_token']
        eos_token = self.config_dict['eos_token']
        pad_token = self.config_dict['pad_token']

        pred_words = []
        tokens[mask_index] = unk_token
        tokens = [sos_token] + tokens + [eos_token]
        indexed_sentence = [self.stoi[token] if token in self.stoi else self.stoi[unk_token] for token in tokens]
        input_tokens = torch.tensor(indexed_sentence, dtype=torch.long, device=device).unsqueeze(0)
        topv, topi = self.model.run_inference(input_tokens, target=None, target_pos=mask_index, k=k)
        for value, key in zip(topv, topi):
            score = value.item()
            word = self.itos[key.item()]
            if word in [unk_token, sos_token, eos_token, pad_token]:
                continue
            pred_words.append((word, score))
        return pred_words

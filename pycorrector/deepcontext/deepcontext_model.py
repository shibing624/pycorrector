# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: context to vector network
"""

import math

import torch
import torch.nn as nn
from loguru import logger
import os
import sys
import time
import operator
import os
import sys
import time

import numpy as np
import torch
from torch import optim

from pycorrector.deepcontext.data_reader import write_config,Dataset
from pycorrector.deepcontext.deepcontext_utils import NegativeSampling, Context2vec
from pycorrector.deepcontext.data_reader import read_config, load_word_dict
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.tokenizer import split_text_into_sentences_by_length
from pycorrector.corrector import Corrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pwd_path = os.path.abspath(os.path.dirname(__file__))



# Training data path.
# chinese corpus
cged_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

sighan_train_path = os.path.join(pwd_path, '../data/cn/sighan_2015/train.tsv')

use_segment = True
segment_type = 'char'
dataset = 'sighan'  # 'sighan' or 'cged'

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
model_dir = os.path.join(output_dir, 'models')
model_path = os.path.join(model_dir, 'model.pth')
vocab_path = os.path.join(model_dir, 'vocab.txt')

# nets
word_embed_size = 200
hidden_size = 200
n_layers = 2
dropout = 0.0

# train
epochs = 20
batch_size = 64
min_freq = 1
learning_rate = 1e-3

if not os.path.exists(model_dir):
    os.makedirs(model_dir)



class DeepContextModel(Corrector):
    def __init__(self,model_dir, vocab_path):
        super(DeepContextModel, self).__init__()
        t1 = time.time()
        # device
        logger.debug("device: {}".format(device))
        model, config_dict = self.load_model(model_dir)
        # norm weight
        model.norm_embedding_weight(model.criterion.W)
        self.model = model
        self.model.eval()

        self.unk_token, self.sos_token, self.eos_token, self.pad_token, self.itos, self.stoi = self._get_config_data(
            config_dict, vocab_path)
        self.model_dir = model_dir
        self.vocab_path = vocab_path
        self.mask = "[]"
        logger.debug('Loaded deep context model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))


    def train_model(self,train_path,
          model_dir,
          vocab_path,
          batch_size=64,
          epochs=3,
          word_embed_size=200,
          hidden_size=200,
          learning_rate=0.0001,
          n_layers=1,
          min_freq=1,
          dropout=0.0):
        print("device: {}".format(device))
        if not os.path.isfile(train_path):
            raise FileNotFoundError

        print('Loading input file')
        dataset = Dataset(train_path,
                          batch_size,
                          min_freq,
                          device,
                          vocab_path)
        counter = np.array([dataset.word_freqs[word] for word in dataset.vocab_2_ids])
        model = Context2vec(vocab_size=len(dataset.vocab_2_ids),
                            counter=counter,
                            word_embed_size=word_embed_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            use_mlp=True,
                            dropout=dropout,
                            pad_index=dataset.pad_index,
                            device=device,
                            is_inference=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print('batch_size:', batch_size, 'epochs:', epochs, 'word_embed_size:', word_embed_size, 'hidden_size:',
              hidden_size, 'device:', device)
        print('model:', model)

        # save model config
        output_config_file = os.path.join(model_dir, 'config.json')
        write_config(output_config_file,
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
        print("train start...")
        for epoch in range(epochs):
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
                    print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'
                          .format(word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                    next_count += interval
                    cur_at = now
                    last_accum_loss = float(total_loss)
                    last_word_count = word_count

            # find best model
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)
            print('epoch:[{}/{}], total_loss:[{}], best_cur_loss:[{}]'
                  .format(epoch + 1, epochs, total_loss.item(), best_loss))
            if is_best:
                torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, 'model_optimizer.pth'))
                print('epoch:{}, save new bert model:{}'.format(epoch + 1, model_dir))

    @staticmethod
    def load_model(model_dir):
        config_file = os.path.join(model_dir, 'config.json')
        config_dict = read_config(config_file)
        model = Context2vec(
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
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
        optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'model_optimizer.pth')))
        return model, config_dict

    @staticmethod
    def _get_config_data(config_dict, vocab_path):
        # load model
        unk_token = config_dict['unk_token']
        sos_token = config_dict['sos_token']
        eos_token = config_dict['eos_token']
        pad_token = config_dict['pad_token']

        # read vocab
        stoi = load_word_dict(vocab_path)
        itos = {v: k for k, v in stoi.items()}

        return unk_token, sos_token, eos_token, pad_token, itos, stoi

    def predict_mask_token(self, tokens, mask_index, k=10):
        pred_words = []
        tokens[mask_index] = self.unk_token
        tokens = [self.sos_token] + tokens + [self.eos_token]
        indexed_sentence = [self.stoi[token] if token in self.stoi else self.stoi[self.unk_token] for token in tokens]
        input_tokens = torch.tensor(indexed_sentence, dtype=torch.long, device=device).unsqueeze(0)
        topv, topi = self.model.run_inference(input_tokens, target=None, target_pos=mask_index, k=k)
        for value, key in zip(topv, topi):
            score = value.item()
            word = self.itos[key.item()]
            if word in [self.unk_token, self.sos_token, self.eos_token, self.pad_token]:
                continue
            pred_words.append((word, score))
        return pred_words

    def predict(self, text, **kwargs):
        details = []
        text_new = ''
        self.check_corrector_initialized()
        # 长句切分为短句
        blocks = split_text_into_sentences_by_length(text, 128)
        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):
                # 处理中文错误
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    sentence_lst[idx] = self.mask
                    # 预测，默认取top10
                    predict_words = self.predict_mask_token(sentence_lst, idx, k=10)
                    top_tokens = []
                    for w, _ in predict_words:
                        top_tokens.append(w)

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append((s, token_str, start_idx + idx, start_idx + idx + 1))
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details
    def eval_model(self):
        pass



    def save_model(self):
        logger.info(f"Saving model into {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))

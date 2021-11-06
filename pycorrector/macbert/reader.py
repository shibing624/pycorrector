# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import os

import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


class CscDataset(Dataset):
    def __init__(self, fp):
        with open(fp, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


def make_loaders(collate_fn, train_path='', valid_path='', test_path='',
                 batch_size=32, test_batch_size=8, num_workers=4):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(CscDataset(train_path),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    valid_loader = None
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(CscDataset(valid_path),
                                  batch_size=test_batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(CscDataset(test_path),
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader

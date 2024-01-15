# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class DataCollator:
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer(t, return_offsets_mapping=True, add_special_tokens=False) for t in ori_texts]
        max_len = max([len(t['input_ids']) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()

        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            off_mapping = encoded_text['offset_mapping']
            for idx in wrong_ids:
                for j, (b, e) in enumerate(off_mapping):
                    if b <= idx < e:
                        # j+1是因为前面的 CLS token
                        det_labels[i, j + 1] = 1
                        break

        return list(ori_texts), list(cor_texts), det_labels


class CscDataset(Dataset):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


def make_loaders(collate_fn, train_path='', valid_path='', test_path='',
                 batch_size=32, num_workers=4):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(
            CscDataset(train_path),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    valid_loader = None
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(
            CscDataset(valid_path),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(
            CscDataset(test_path),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    return train_loader, valid_loader, test_loader

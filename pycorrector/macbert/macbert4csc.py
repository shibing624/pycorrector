# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
from abc import ABC

import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from pycorrector.macbert.base_model import CscTrainingModel, FocalLoss


class MacBert4Csc(CscTrainingModel, ABC):
    def __init__(self, tokenizer, lr=5e-5, weight_decay=0.01, optimizer_name='AdamW',
                 loss_coefficient=0.3, device=torch.device('cuda'),
                 pretrained_model='hfl/chinese-macbert-base', *args, **kwargs):
        super().__init__(lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name,
                         loss_coefficient=loss_coefficient, device=device, *args, **kwargs)
        self.bert = BertForMaskedLM.from_pretrained(pretrained_model)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels[text_labels == 0] = -100
            # text_labels = text_labels.to(self._device)
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        # encoded_text.to(self._device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')  # nn.BCELoss()
            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)
        return outputs

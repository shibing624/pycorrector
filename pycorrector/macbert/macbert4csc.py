# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import operator
import torch
import torch.nn as nn
import numpy as np
from transformers import BertForMaskedLM
import pytorch_lightning as pl
from pycorrector.macbert import lr_scheduler
from pycorrector.macbert.evaluate_util import compute_corrector_prf, compute_sentence_level_prf
from pycorrector.utils.logger import logger


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


def make_optimizer(lr, weight_decay, optimizer_name, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            lr = lr * 2
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, optimizer_name)(params)
    return optimizer


def build_lr_scheduler(optimizer):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": 0.01,
        "warmup_epochs": 1024,
        "warmup_method": "linear",

        # multi-step lr scheduler options
        "milestones": (10,),
        "gamma": 0.9999,

        # cosine annealing lr scheduler options
        "max_iters": 10,
        "delay_iters": 0,
        "eta_min_lr": 3e-7,

    }
    scheduler = getattr(lr_scheduler, "WarmupExponentialLR")(**scheduler_args)
    return {'scheduler': scheduler, 'interval': 'step'}


class BaseModel(pl.LightningModule):
    """用于CSC的BaseModel, 定义了训练及预测步骤"""

    def __init__(self, loss_coefficient=0.3, device=torch.device('cuda'), lr=5e-5,
                 weight_decay=0.01, optimizer_name='AdamW', *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        # loss weight
        self.w = loss_coefficient
        self._device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        return loss

    def configure_optimizers(self):
        optimizer = make_optimizer(self.lr, self.weight_decay, self.optimizer_name, self)
        scheduler = build_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_validation_epoch_start(self) -> None:
        logger.info('Valid.')

    def on_test_epoch_start(self) -> None:
        logger.info('Testing...')

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
            results.append((_src, _tgt, _predict,))

        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def validation_epoch_end(self, outputs) -> None:
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
        loss = np.mean([out[0] for out in outputs])
        logger.info('val_loss', loss)
        logger.info(f'loss: {loss}')
        logger.info(f'Detection:\n'
                    f'acc: {np.mean(det_acc_labels):.4f}')
        logger.info(f'Correction:\n'
                    f'acc: {np.mean(cor_acc_labels):.4f}')
        compute_corrector_prf(results, logger)
        compute_sentence_level_prf(results, logger)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        logger.info('Test.')
        self.validation_epoch_end(outputs)

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self._device)
        with torch.no_grad():
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        return rst


class MacBert4Csc(BaseModel):
    def __init__(self, tokenizer, loss_coefficient=0.3, device=torch.device('cuda'),
                 pretrained_model='hfl/chinese-macbert-base'):
        super(MacBert4Csc, self).__init__(loss_coefficient, device)
        self.bert = BertForMaskedLM.from_pretrained(pretrained_model)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self._device)
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self._device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
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

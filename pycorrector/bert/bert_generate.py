# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

refer https://github.com/voidful/BertGenerate/blob/master/Bert_Generate.ipynb
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from pycorrector import config

input_text = "[CLS] I go to school by bus [SEP] "
target_text = "我搭校车上学"
modelpath = config.bert_model_dir
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = BertForMaskedLM.from_pretrained(modelpath)


# cuda
# model.to('cuda')

def get_example_pair(input_text, target_text):
    example_pair = dict()

    for i in range(0, len(target_text) + 1):
        tokenized_text = tokenizer.tokenize(input_text)
        tokenized_text.extend(target_text[:i])
        tokenized_text.append('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')

        loss_ids = [-1] * (len(tokenizer.convert_tokens_to_ids(tokenized_text)) - 1)
        if i == len(target_text):
            loss_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[SEP]'))[0])
        else:
            loss_ids.append(tokenizer.convert_tokens_to_ids([target_text[i]])[0])
        loss_tensors = torch.tensor([loss_ids]).to('cpu')

        example_pair[tokens_tensor] = loss_tensors
        print(tokenized_text, loss_ids, loss_ids[-1])
    # print(example_pair)
    return example_pair


class BertLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # modelpath = "bert-base-chinese"
        modelpath = config.bert_model_dir
        self.bert = BertModel.from_pretrained(modelpath)
        self.rnn = nn.LSTM(num_layers=2, dropout=0.2, input_size=768, hidden_size=768 // 2)
        self.fc = nn.Linear(384, self.bert.config.vocab_size)

    def forward(self, x, y=None):
        self.bert.train()
        encoded_layers = self.bert(x)
        enc, _ = self.rnn(encoded_layers.last_hidden_state)
        logits = self.fc(enc)
        if y is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), y.view(-1))
            return loss
        return logits


model = BertLSTM()
optimizer = torch.optim.Adamax(model.parameters(), lr=5e-5)

train_example_pair = get_example_pair(input_text, target_text)
model.train()
for i in range(0, 10):
    eveloss = 0
    for k, v in train_example_pair.items():
        optimizer.zero_grad()
        loss = model(k, v)
        eveloss += loss.mean().item()
        loss.backward()
        optimizer.step()
    print("step " + str(i) + " : " + str(eveloss))


test_input_text = "[CLS] I want go to school[SEP] "
test_target_text = "我想去上学"
test_example_pair = get_example_pair(test_input_text, test_target_text)

model.eval()
for k, v in test_example_pair.items():
    predictions = model(k)
    predicted_index = torch.argmax(predictions[0, -1]).item()
    if predicted_index < model.bert.config.vocab_size:
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)

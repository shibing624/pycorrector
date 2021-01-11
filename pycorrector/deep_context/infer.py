# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: Inference
"""
import sys

import torch
from torch import optim

sys.path.append('../..')
from pycorrector.deep_context.model import Context2vec
from pycorrector.deep_context import config
from pycorrector.deep_context.data_util import read_config, load_vocab


def read_model(model_path, device):
    config_file = model_path + '.config.json'
    config_dict = read_config(config_file)
    model = Context2vec(vocab_size=config_dict['vocab_size'],
                        counter=[1] * config_dict['vocab_size'],
                        word_embed_size=config_dict['word_embed_size'],
                        hidden_size=config_dict['hidden_size'],
                        n_layers=config_dict['n_layers'],
                        bidirectional=config_dict['bidirectional'],
                        use_mlp=config_dict['use_mlp'],
                        dropout=config_dict['dropout'],
                        pad_index=config_dict['pad_index'],
                        device=device,
                        inference=True
                        ).to(device)
    model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
    optimizer.load_state_dict(torch.load(model_path + '_optim'))
    model.eval()
    return model, config_dict


def get_infer_data(model_path,
                   emb_path,
                   gpu_id,
                   ):
    # device
    device = torch.device('cpu')
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
        print("use gpu, gpu_id={}".format(gpu_id))

    # load model
    model, config_dict = read_model(model_path, device)
    unk_token = config_dict['unk_token']
    bos_token = config_dict['bos_token']
    eos_token = config_dict['eos_token']
    pad_token = config_dict['pad_token']

    # read vocab from word_emb path
    itos, stoi = load_vocab(emb_path)

    # norm weight
    model.norm_embedding_weight(model.criterion.W)
    return model, unk_token, bos_token, eos_token, pad_token, itos, stoi, device


def return_split_sentence(sentence):
    if ' ' not in sentence:
        print('sentence should contain white space to split it into tokens')
        raise SyntaxError
    elif '[]' not in sentence:
        print('sentence should contain `[]` that notes the target')
        raise SyntaxError
    else:
        tokens = sentence.lower().strip().split()
        target_pos = tokens.index('[]')
        return tokens, target_pos


def infer_one_sentence(sentence, model, unk_token, bos_token, eos_token, pad_token, itos, stoi, device):
    pred_words = []
    tokens, target_pos = [], 0
    try:
        tokens, target_pos = return_split_sentence(sentence)
    except SyntaxError:
        pass
    tokens[target_pos] = unk_token
    tokens = [bos_token] + tokens + [eos_token]
    indexed_sentence = [stoi[token] if token in stoi else stoi[unk_token] for token in tokens]
    input_tokens = torch.tensor(indexed_sentence, dtype=torch.long, device=device).unsqueeze(0)
    topv, topi = model.run_inference(input_tokens, target=None, target_pos=target_pos, k=10)
    for value, key in zip(topv, topi):
        score = value.item()
        word = itos[key.item()]
        if word in [unk_token, bos_token, eos_token, pad_token]:
            continue
        pred_words.append((word, score))
    return pred_words


if __name__ == "__main__":
    sents = ["而 且 我 希 望 不 再 存 在 抽 [] 的 人 。",
             "男 女 分 班 的 问 题 有 什 [] 好 处 ？",
             "由 我 开 始 [] 起 。"]
    model, unk_token, bos_token, eos_token, pad_token, itos, stoi, device = get_infer_data(config.model_path,
                                                                                           config.emb_path,
                                                                                           config.gpu_id)
    for i in sents:
        r = infer_one_sentence(i, model, unk_token, bos_token, eos_token, pad_token, itos, stoi, device)
        print(i, r)

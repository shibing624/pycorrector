# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

import numpy as np
import torch

sys.path.append('../..')

from pycorrector.seq2seq import config
from pycorrector.seq2seq.data_reader import SOS_TOKEN, EOS_TOKEN
from pycorrector.seq2seq.data_reader import load_word_dict
from pycorrector.seq2seq.seq2seq import Seq2Seq
from pycorrector.seq2seq.convseq2seq import ConvSeq2Seq
from pycorrector.seq2seq.data_reader import PAD_TOKEN
from pycorrector.seq2seq.seq2seq_model import Seq2SeqModel
from pycorrector.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference(object):
    def __init__(self, arch, model_dir, src_vocab_path, trg_vocab_path,
                 embed_size=50, hidden_size=50, dropout=0.5, max_length=128):
        if arch == 'bert2seq2seq':
            # Bert Seq2seq model
            print('use bert seq2seq model.')
            use_cuda = True if torch.cuda.is_available() else False

            # encoder_type=None, encoder_name=None, decoder_name=None
            self.model = Seq2SeqModel("bert", "{}/encoder".format(model_dir),
                                      "{}/decoder".format(model_dir), use_cuda=use_cuda)
        elif arch in ['seq2seq', 'convseq2seq']:
            self.src_2_ids = load_word_dict(src_vocab_path)
            self.trg_2_ids = load_word_dict(trg_vocab_path)
            self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
            if arch == 'seq2seq':
                print('use seq2seq model.')
                self.model = Seq2Seq(encoder_vocab_size=len(self.src_2_ids),
                                     decoder_vocab_size=len(self.trg_2_ids),
                                     embed_size=embed_size,
                                     enc_hidden_size=hidden_size,
                                     dec_hidden_size=hidden_size,
                                     dropout=dropout).to(device)
                model_path = os.path.join(model_dir, 'seq2seq.pth')
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            else:
                print('use convseq2seq model.')
                trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
                self.model = ConvSeq2Seq(encoder_vocab_size=len(self.src_2_ids),
                                         decoder_vocab_size=len(self.trg_2_ids),
                                         embed_size=embed_size,
                                         enc_hidden_size=hidden_size,
                                         dec_hidden_size=hidden_size,
                                         dropout=dropout,
                                         trg_pad_idx=trg_pad_idx,
                                         device=device,
                                         max_length=max_length).to(device)
                model_path = os.path.join(model_dir, 'convseq2seq.pth')
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
        else:
            logger.error('error arch: {}'.format(arch))
            raise ValueError("Model arch choose error. Must use one of seq2seq model.")
        self.arch = arch
        self.max_length = max_length

    def predict(self, sentence_list):
        result = []
        if self.arch in ['seq2seq', 'convseq2seq']:
            for query in sentence_list:
                out = []
                tokens = [token.lower() for token in query]
                tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
                src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]

                sos_idx = self.trg_2_ids[SOS_TOKEN]
                if self.arch == 'seq2seq':
                    src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
                    src_tensor_len = torch.from_numpy(np.array([len(src_ids)])).long().to(device)
                    sos_tensor = torch.Tensor([[self.trg_2_ids[SOS_TOKEN]]]).long().to(device)
                    translation, attn = self.model.translate(src_tensor, src_tensor_len, sos_tensor, self.max_length)
                    translation = [self.id_2_trgs[i] for i in translation.data.cpu().numpy().reshape(-1) if
                                   i in self.id_2_trgs]
                else:
                    src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
                    translation, attn = self.model.translate(src_tensor, sos_idx)
                    translation = [self.id_2_trgs[i] for i in translation if i in self.id_2_trgs]
                for word in translation:
                    if word != EOS_TOKEN:
                        out.append(word)
                    else:
                        break
                result.append(''.join(out))
        elif self.arch == 'bertseq2seq':
            corrected_sents = self.model.predict(sentence_list)
            result = [i.replace(' ', '') for i in corrected_sents]
        else:
            raise ValueError('error arch.')
        return result


if __name__ == "__main__":
    m = Inference(config.arch,
                  config.model_dir,
                  config.src_vocab_path,
                  config.trg_vocab_path,
                  embed_size=config.embed_size,
                  hidden_size=config.hidden_size,
                  dropout=config.dropout,
                  max_length=config.max_length
                  )
    inputs = [
        '老是较书。',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。'
    ]
    outputs = m.predict(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b)
        print()
# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

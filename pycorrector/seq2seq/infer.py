# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import operator
import numpy as np
import torch
import argparse
sys.path.append('../..')

from pycorrector.seq2seq.data_reader import SOS_TOKEN, EOS_TOKEN
from pycorrector.seq2seq.data_reader import load_word_dict
from pycorrector.seq2seq.seq2seq import Seq2Seq
from pycorrector.seq2seq.convseq2seq import ConvSeq2Seq
from pycorrector.seq2seq.data_reader import PAD_TOKEN
from pycorrector.seq2seq.seq2seq_model import Seq2SeqModel
from pycorrector.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '《', '》']


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class Inference(object):
    def __init__(self, model_dir, arch='convseq2seq',
                 embed_size=128, hidden_size=128, dropout=0.25, max_length=128):
        logger.debug("device: {}".format(device))
        if arch in ['seq2seq', 'convseq2seq']:
            src_vocab_path = os.path.join(model_dir, 'vocab_source.txt')
            trg_vocab_path = os.path.join(model_dir, 'vocab_target.txt')
            self.src_2_ids = load_word_dict(src_vocab_path)
            self.trg_2_ids = load_word_dict(trg_vocab_path)
            self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
            if arch == 'seq2seq':
                logger.debug('use seq2seq model.')
                self.model = Seq2Seq(encoder_vocab_size=len(self.src_2_ids),
                                     decoder_vocab_size=len(self.trg_2_ids),
                                     embed_size=embed_size,
                                     enc_hidden_size=hidden_size,
                                     dec_hidden_size=hidden_size,
                                     dropout=dropout).to(device)
                model_path = os.path.join(model_dir, 'seq2seq.pth')
                logger.debug('load model from {}'.format(model_path))
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()
            else:
                logger.debug('use convseq2seq model.')
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
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                logger.debug('load model from {}'.format(model_path))
                self.model.eval()
        elif arch == 'bertseq2seq':
            # Bert Seq2seq model
            logger.debug('use bert seq2seq model.')
            use_cuda = True if torch.cuda.is_available() else False

            # encoder_type=None, encoder_name=None, decoder_name=None
            self.model = Seq2SeqModel("bert", "{}/encoder".format(model_dir),
                                      "{}/decoder".format(model_dir), use_cuda=use_cuda)
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
                corrected_text = ''.join(out)
                corrected_text, sub_details = get_errors(corrected_text, query)
                result.append([corrected_text, sub_details])
        else:
            corrected_sents = self.model.predict(sentence_list)
            result = [i.replace(' ', '') for i in corrected_sents]
            for c, s in zip(corrected_sents, sentence_list):
                c = c.replace(' ', '')
                c, sub_details = get_errors(c, s)
                result.append([c, sub_details])
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="output/sighan_convseq2seq/", type=str, help="Dir for model save.")
    parser.add_argument("--arch",
                        default="convseq2seq", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(
                            ['seq2seq', 'convseq2seq', 'bertseq2seq']),
                        )
    parser.add_argument("--max_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, sequences shorter padded.",
                        )
    parser.add_argument("--embed_size", default=128, type=int, help="Embedding size.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout rate.")

    args = parser.parse_args()
    print(args)

    m = Inference(args.model_dir,
                  args.arch,
                  embed_size=args.embed_size,
                  hidden_size=args.hidden_size,
                  dropout=args.dropout,
                  max_length=args.max_length
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
        print('predict:', b[0], b[1])
        print()

# result:
# input  : 老是较书。
# predict: 老师教书。 [('是', '师', 1, 2), ('较', '教', 2, 3)]
#
# input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
# predict: 感谢等五分以后，碰到一位很棒的女生跟我可聊。 [('奴', '女', 15, 16)]

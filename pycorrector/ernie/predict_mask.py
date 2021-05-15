# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import numpy as np
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L

sys.path.append('../..')
from pycorrector.ernie.modeling_ernie import ErnieModelForPretraining, ErnieModel
from pycorrector.ernie.tokenizing_ernie import ErnieTokenizer

model_dir = 'ernie-1.0'

D.guard().__enter__() # activate paddle `dygrpah` mode
tokenizer = ErnieTokenizer.from_pretrained(model_dir)
rev_dict = {v: k for k, v in tokenizer.vocab.items()}
rev_dict[tokenizer.pad_id] = ''  # replace [PAD]
rev_dict[tokenizer.sep_id] = ''  # replace [PAD]
rev_dict[tokenizer.unk_id] = ''  # replace [PAD]


class ErnieCloze(ErnieModelForPretraining):
    def __init__(self, *args, **kwargs):
        super(ErnieCloze, self).__init__(*args, **kwargs)
        del self.pooler_heads

    def forward(self, src_ids, *args, **kwargs):
        pooled, encoded = ErnieModel.forward(self, src_ids, *args, **kwargs)
        encoded_2d = L.gather_nd(encoded, L.where(src_ids == mask_id))
        encoded_2d = self.mlm(encoded_2d)
        encoded_2d = self.mlm_ln(encoded_2d)
        logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
        return logits_2d


@np.vectorize
def rev_lookup(i):
    return rev_dict[i]


ernie = ErnieCloze.from_pretrained(model_dir)
ernie.eval()

ids, _ = tokenizer.encode('戊[MASK]变法，又称百日维新，是 [MASK] [MASK] [MASK] 、梁启超等维新派人士通过光绪帝进行 的一场资产阶级改良。')
mask_id = tokenizer.mask_id
print(ids)
ids = np.expand_dims(ids, 0)
ids = D.to_variable(ids)
logits = ernie(ids).numpy()
output_ids = np.argmax(logits, -1)
seg_txt = rev_lookup(output_ids)

print(''.join(seg_txt))


def predict_mask(sentence_with_mask):
    """
    predict multi masks, support top5, multi mask
    :param sentence_with_mask:
    :return:
    """
    ids, id_types = tokenizer.encode(sentence_with_mask)
    mask_id = tokenizer.mask_id
    # print(ids, id_types, mask_id)
    ids = np.expand_dims(ids, 0)
    ids = D.to_variable(ids)
    logits = ernie(ids).numpy()
    output_ids = np.argsort(logits, -1)
    j_ret = []
    for i in output_ids:
        i_ret = []
        for j in i[::-1][:5]:
            i_ret.append(rev_dict[j])
        j_ret.append(i_ret)
    out = []
    for i in range(len(j_ret)):
        temp = []
        for j in range(len(j_ret[i])):
            temp.append(j_ret[i][j])
        out.append(temp)
    print(out)
    out = np.array(out)
    out = np.transpose(out).tolist()
    print(' '.join([''.join(i) for i in out]))
    return out


i = predict_mask('机器学习是人工[MASK]能领遇最能体现智能的一个分知')
print(i)

i = predict_mask('hi lili, What is the name of the [MASK] ?')
print(i)

i = predict_mask('今天[MASK]情很[MASK]')
print(i)

i = predict_mask('少先队员[MASK]该为老人让座')
print(i)

i = predict_mask('[MASK]七学习是人工智能领遇最能体现智能的一个分知')
print(i)

i = predict_mask('机[MASK]学习是人工[MASK][MASK]领遇最能体现智能的一个分知')
print(i)

i = predict_mask('机器学习是人工[MASK][MASK]领遇最能体现智能的一个分知')
print(i)

print(predict_mask('机[MASK]学习是人工智能领遇最能体现[MASK][MASK]的一个[MASK][MASK]'))

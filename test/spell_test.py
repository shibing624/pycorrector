# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from pycorrector.spell import *
from pycorrector.util import segment
import pycorrector.spell

# test kenlm
trigram_path = os.path.join(pwd_path, '../pycorrector/data/kenlm/people_words.klm')
trigram = kenlm.Model(trigram_path)
print('Loaded trigram_word language model from {}'.format(trigram_path))


def get_ngram_score(chars, model=trigram):
    score = model.score(' '.join(chars), bos=False, eos=False)
    # print('score: {}'.format(round(score, 4)))
    score_raw = model.score(' '.join(chars))
    ppl = model.perplexity(' '.join(chars))
    return score, score_raw, ppl


def lm_score(sentence):
    sent_seg = segment(sentence)
    sc, sc_r, ppl = get_ngram_score(sent_seg)
    print('score:%s,score_raw:%s,ppl:%s' % (sc, sc_r, ppl))


sent1 = '少先队员因该为老人让坐'
sent2 = '少先队员应该为老人让座'

lm_score(sent1)
lm_score(sent2)

print('zhwiki:',pycorrector.spell.get_ngram_score(segment(sent1),pycorrector.spell.bigram))
print('zhwiki:',pycorrector.spell.get_ngram_score(segment(sent2),pycorrector.spell.bigram))

sent1 = '老人让坐'
sent2 = '老人让座'

lm_score(sent1)
lm_score(sent2)



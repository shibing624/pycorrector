# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import operator
import sys
import time
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch

sys.path.append('../..')
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector import config
from pycorrector.utils.tokenizer import split_text_by_maxlen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '']


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
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class MacBertCorrector(object):
    def __init__(self, model_dir=config.macbert_model_dir):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        t1 = time.time()
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/macbert4csc-base-chinese"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForMaskedLM.from_pretrained(model_dir)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def macbert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        block_texts = [block[0] for block in blocks]
        inputs = self.tokenizer(block_texts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for ids, (text, idx) in zip(outputs.logits, blocks):
            decode_tokens = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            details.extend(sub_details)
        return text_new, details


if __name__ == "__main__":
    m = MacBertCorrector()
    error_sentences = [
        '内容提要——在知识产权学科领域里',
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
        '他们的吵翻很不错，再说他们做的咖喱鸡也好吃',
        '影像小孩子想的快，学习管理的斑法',
        '餐厅的换经费产适合约会',
        '走路真的麻坊，我也没有喝的东西，在家汪了',
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '妈妈说："别趴地上了，快起来，你还吃饭吗？"，我说："好。"就扒起来了。',
        '你说：“怎么办？”我怎么知道？',
        '我父母们常常说：“那时候吃的东西太少，每天只能吃一顿饭。”想一想，人们都快要饿死，谁提出化肥和农药的污染。',
        '这本新书《居里夫人传》将的很生动有趣',
        '֍我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……֍新的食谱',
        '注意：“跨类保护”不等于“全类保护”。',
        '12.——对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性；',
        '《著作权法》（2020修正）第23条：“自然人的作品，其发表权、本法第',
        '三步检验法（三步检验标准）（three-step test）：若要',
        '①申请人应提交，顺应了国家“健全创新激励',
        '①申请人应提交,太平天国领导人洪仁玕。',
        '	部分优先权：',
        '实施其专利的行为（生产经营≠营利≠商业经营）',
        '实施,i can speak chinese, can i spea english. ? hello.',
    ]
    for sent in error_sentences:
        corrected_sent, err = m.macbert_correct(sent)
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))

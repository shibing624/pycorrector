# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import operator
import sys
import time
import os
import jieba
from transformers import BertTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch
from typing import List
from loguru import logger

sys.path.append('../..')
from pycorrector import config
from pycorrector.utils.tokenizer import split_text_by_maxlen
from pycorrector.utils.text_utils import is_chinese

jieba.setLogLevel('ERROR')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '擤', '玕']


class ZHTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


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


class CopyT5Corrector(object):
    def __init__(self, model_dir=config.copyt5_model_dir):
        self.name = 'copyt5_corrector'
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/copyt5-base-chinese-correction"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')
        t1 = time.time()
        self.tokenizer = ZHTokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def t5_correct(self, text: str, max_length: int = 128):
        """
        句子纠错
        :param text: str, sentence
        :param max_length: int, max length of each sentence
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []

        blocks = split_text_by_maxlen(text, maxlen=max_length)
        block_texts = [block[0] for block in blocks]
        inputs = self.tokenizer(block_texts, padding=True, max_length=max_length, truncation=True,
                                return_tensors='pt').to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.sep_token_id,
                decoder_start_token_id=self.tokenizer.cls_token_id,
                num_beams=3,
                max_length=max_length,
                num_return_sequences=1
            )
            outputs = outputs[:, 1:].cpu().numpy()
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds = [s.replace(' ', '') for s in preds]
        for i, (text, idx) in enumerate(blocks):
            corrected_text = preds[i]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            details.extend(sub_details)
        return text_new, details

    def batch_t5_correct(self, texts: List[str], max_length: int = 128, batch_size: int = 256, silent: bool = False):
        """
        句子纠错
        :param texts: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, bz
        :param silent: bool, show log
        :return: list, (corrected_text, [error_word, correct_word, begin_pos, end_pos])
        """
        result = []
        for batch in tqdm([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)],
                          desc="Generating outputs", disable=silent):
            inputs = self.tokenizer(batch, padding=True, max_length=max_length, truncation=True,
                                    return_tensors='pt').to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.sep_token_id,
                    decoder_start_token_id=self.tokenizer.cls_token_id,
                    num_beams=3,
                    max_length=max_length,
                    num_return_sequences=1
                )
                outputs = outputs[:, 1:].cpu().numpy()
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds = [s.replace(' ', '') for s in preds]
            for i, text in enumerate(batch):
                text_new = ''
                details = []
                idx = 0
                corrected_text = preds[i]
                corrected_text, sub_details = get_errors(corrected_text, text)
                text_new += corrected_text
                sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
                details.extend(sub_details)
                result.append([text_new, details])
        return result


if __name__ == "__main__":
    m = CopyT5Corrector('./copyt5-base-chinese-correction/')
    error_sentences = [
        '各省自治区直辖市人民政府，国院各部委各直属机构，',
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
        '三步检验法“三步‘检验’标准”（three-step test）：若要',
    ]
    t1 = time.time()
    for sent in error_sentences:
        corrected_sent, err = m.t5_correct(sent)
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))
    print('[single]spend time:', time.time() - t1)
    t2 = time.time()
    res = m.batch_t5_correct(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} err:{}".format(sent, r[0], r[1]))
    print('[batch]spend time:', time.time() - t2)

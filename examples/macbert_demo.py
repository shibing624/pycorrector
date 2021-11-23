# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import operator

sys.path.append("..")
from pycorrector.macbert.macbert_corrector import MacBertCorrector


def use_origin_transformer():
    import torch
    from transformers import BertTokenizer, BertForMaskedLM

    tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
    model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")

    texts = ["今天新情很好", "你找到你最喜欢的工作，我也很高心。"]
    outputs = model(**tokenizer(texts, padding=True, return_tensors='pt'))

    def get_errors(corrected_text, origin_text):
        details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊']:
                # add blank space
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                details.append((ori_char, corrected_text[i], i, i + 1))
        details = sorted(details, key=operator.itemgetter(2))
        return corrected_text, details

    result = []
    for ids, text in zip(outputs.logits, texts):
        _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
        corrected_text = _text[:len(text)]
        corrected_text, details = get_errors(corrected_text, text)
        print(text, ' => ', corrected_text, details)
        result.append((corrected_text, details))
    print(result)
    return result


if __name__ == '__main__':
    use_origin_transformer()
    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
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
    ]

    m = MacBertCorrector()
    for line in error_sentences:
        correct_sent, err = m.macbert_correct(line)
        print("query:{} => {} err:{}".format(line, correct_sent, err))

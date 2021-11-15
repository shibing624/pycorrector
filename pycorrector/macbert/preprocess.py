# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com), abtion{at}outlook.com
# Brief:
import os
import sys
from codecs import open
from sklearn.model_selection import train_test_split
from lxml import etree

sys.path.append('../..')
from pycorrector.utils.io_utils import save_json
from pycorrector import traditional2simplified

pwd_path = os.path.abspath(os.path.dirname(__file__))


def read_data(fp):
    for fn in os.listdir(fp):
        if fn.endswith('ing.sgml'):
            with open(os.path.join(fp, fn), 'r', encoding='utf-8', errors='ignore') as f:
                item = []
                for line in f:
                    if line.strip().startswith('<ESSAY') and len(item) > 0:
                        yield ''.join(item)
                        item = [line.strip()]
                    elif line.strip().startswith('<'):
                        item.append(line.strip())


def proc_item(item):
    """
    处理训练数据集
    Args:
        item:
    Returns:
        list
    """
    root = etree.XML(item)
    passages = dict()
    mistakes = []
    for passage in root.xpath('/ESSAY/TEXT/PASSAGE'):
        passages[passage.get('id')] = traditional2simplified(passage.text)
    for mistake in root.xpath('/ESSAY/MISTAKE'):
        mistakes.append({'id': mistake.get('id'),
                         'location': int(mistake.get('location')) - 1,
                         'wrong': traditional2simplified(mistake.xpath('./WRONG/text()')[0].strip()),
                         'correction': traditional2simplified(mistake.xpath('./CORRECTION/text()')[0].strip())})

    rst_items = dict()

    def get_passages_by_id(pgs, _id):
        p = pgs.get(_id)
        if p:
            return p
        _id = _id[:-1] + str(int(_id[-1]) + 1)
        p = pgs.get(_id)
        if p:
            return p
        raise ValueError(f'passage not found by {_id}')

    for mistake in mistakes:
        if mistake['id'] not in rst_items.keys():
            rst_items[mistake['id']] = {'original_text': get_passages_by_id(passages, mistake['id']),
                                        'wrong_ids': [],
                                        'correct_text': get_passages_by_id(passages, mistake['id'])}

        # todo 繁体转简体字符数量或位置发生改变校验
        ori_text = rst_items[mistake['id']]['original_text']
        cor_text = rst_items[mistake['id']]['correct_text']
        if len(ori_text) == len(cor_text):
            if ori_text[mistake['location']] in mistake['wrong']:
                rst_items[mistake['id']]['wrong_ids'].append(mistake['location'])
                wrong_char_idx = mistake['wrong'].index(ori_text[mistake['location']])
                start = mistake['location'] - wrong_char_idx
                end = start + len(mistake['wrong'])
                rst_items[mistake['id']][
                    'correct_text'] = f'{cor_text[:start]}{mistake["correction"]}{cor_text[end:]}'
        else:
            print(f'error line:\n{mistake["id"]}\n{ori_text}\n{cor_text}')
    rst = []
    for k in rst_items.keys():
        if len(rst_items[k]['correct_text']) == len(rst_items[k]['original_text']):
            rst.append({'id': k, **rst_items[k]})
        else:
            text = rst_items[k]['correct_text']
            rst.append({'id': k, 'correct_text': text, 'original_text': text, 'wrong_ids': []})
    return rst


def proc_test_set(fp):
    """
    生成sighan15的测试集
    Args:
        fp:
    Returns:
    """
    inputs = dict()
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestInput.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            pid = line[5:14]
            text = line[16:].strip()
            inputs[pid] = text

    rst = []
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestTruth.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            pid = line[0:9]
            mistakes = line[11:].strip().split(', ')
            if len(mistakes) <= 1:
                text = traditional2simplified(inputs[pid])
                rst.append({'id': pid,
                            'original_text': text,
                            'wrong_ids': [],
                            'correct_text': text})
            else:
                wrong_ids = []
                original_text = inputs[pid]
                cor_text = inputs[pid]
                for i in range(len(mistakes) // 2):
                    idx = int(mistakes[2 * i]) - 1
                    cor_char = mistakes[2 * i + 1]
                    wrong_ids.append(idx)
                    cor_text = f'{cor_text[:idx]}{cor_char}{cor_text[idx + 1:]}'
                original_text = traditional2simplified(original_text)
                cor_text = traditional2simplified(cor_text)
                if len(original_text) != len(cor_text):
                    print('error line:\n', pid)
                    print(original_text)
                    print(cor_text)
                    continue
                rst.append({'id': pid,
                            'original_text': original_text,
                            'wrong_ids': wrong_ids,
                            'correct_text': cor_text})

    return rst


def main():
    # 注意：训练样本过少，仅作为模型测试使用
    sighan15_dir = os.path.join(pwd_path, '../data/cn/sighan_2015/')
    rst_items = []
    test_lst = proc_test_set(sighan15_dir)
    for item in read_data(sighan15_dir):
        rst_items += proc_item(item)

    # 拆分训练与测试
    print('data_size:', len(rst_items))
    train_lst, dev_lst = train_test_split(rst_items, test_size=0.1, random_state=42)
    save_json(train_lst, os.path.join(pwd_path, 'output/train.json'))
    save_json(train_lst, os.path.join(pwd_path, 'output/dev.json'))
    save_json(test_lst, os.path.join(pwd_path, 'output/test.json'))


if __name__ == '__main__':
    main()

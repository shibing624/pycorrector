# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import operator

from pycorrector.utils.text_utils import is_chinese_char


def get_errors(corrected_text, origin_text):
    """Get errors between corrected text and origin text"""
    new_corrected_text = ""
    errors = []
    i, j = 0, 0
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    while i < len(origin_text) and j < len(corrected_text):
        if origin_text[i] in unk_tokens:
            new_corrected_text += origin_text[i]
            i += 1
        elif corrected_text[j] in unk_tokens:
            new_corrected_text += corrected_text[j]
            j += 1
        # Deal with Chinese characters
        elif is_chinese_char(origin_text[i]) and is_chinese_char(corrected_text[j]):
            # If the two characters are the same, then the two pointers move forward together
            if origin_text[i] == corrected_text[j]:
                new_corrected_text += corrected_text[j]
                i += 1
                j += 1
            else:
                # Check for insertion errors
                if j + 1 < len(corrected_text) and origin_text[i] == corrected_text[j + 1]:
                    errors.append(('', corrected_text[j], j))
                    new_corrected_text += corrected_text[j]
                    j += 1
                # Check for deletion errors
                elif i + 1 < len(origin_text) and origin_text[i + 1] == corrected_text[j]:
                    errors.append((origin_text[i], '', i))
                    i += 1
                # Check for replacement errors
                else:
                    errors.append((origin_text[i], corrected_text[j], i))
                    new_corrected_text += corrected_text[j]
                    i += 1
                    j += 1
        else:
            new_corrected_text += origin_text[i]
            if origin_text[i] == corrected_text[j]:
                j += 1
            i += 1
    errors = sorted(errors, key=operator.itemgetter(2))
    return new_corrected_text, errors


def get_errors_for_t5(corrected_text, origin_text):
    """Get new corrected text and errors between corrected text and origin text"""
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            if not is_chinese_char(ori_char):
                # pass not chinese char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            if not is_chinese_char(corrected_text[i]):
                corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                continue
            errors.append((ori_char, corrected_text[i], i))
    errors = sorted(errors, key=operator.itemgetter(2))
    return corrected_text, errors


if __name__ == '__main__':
    sentence_pairs = [
        ('内容提要在知识产权学科领域里', '内容提要——在知识产权学科领域里'),
        ('首金得主杨倩、三跳满分的全红婵、举重纪录创造者李雯雯00后选手闪亮奥运舞台。',
         '首金得主杨倩、三跳满分的全红婵、举重纪录创造者李雯雯……“00后”选手闪曜奥运舞台。'),
        ('现在银色的k2p是mtk还是博通啊？', '现在银色的K2P是MTK还是博通啊？'),
        ('u盘有送挂绳吗少先队员因该为老人让坐', 'U盘有送挂绳吗少先队员因该为老人让坐'),
        ('你说：怎么办？我怎么知道？', '你说：“怎么办？”我怎么知道？'),
        ('我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡新的食谱', '֍我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……֍新的食谱'),
        ('12.在对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性',
         '12.——对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性；'),
        ('三步检验法（三步检验标准）（三-steptest）：若要', '三步检验法（三步检验标准）（three-step test）：若要'),
        ('部分优先权：', '	部分优先权：'),
        ('我不想看琅琊榜', '我不唉“看 琅擤琊榜”'),
        ('我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡', '我喜欢吃鸡，公鸡、母鸡、白白切鸡、乌鸡、紫燕鸡'),  # 多字
        ('他德语说的很好', '他德德语说的很好'),  # 多字
        ('他德语说的很好', '他语说的很好'),  # 少字
        ('我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡', '我喜欢吃鸡，公鸡、母鸡、切鸡、乌鸡、紫燕鸡'),  # 少字
    ]
    for pair in sentence_pairs:
        new_corrected_text, errors = get_errors(pair[0], pair[1])
        print(f"{new_corrected_text} {errors}")

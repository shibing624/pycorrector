# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import difflib


def get_errors(corrected_text, origin_text):
    """Get errors between corrected text and origin text"""
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    s = difflib.SequenceMatcher(None, origin_text, corrected_text)
    new_corrected_text = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if origin_text[i] not in unk_tokens and corrected_text[j] not in unk_tokens:
                    errors.append((origin_text[i], corrected_text[j], i))
                new_corrected_text += corrected_text[j]
        elif tag == 'delete':
            for i in range(i1, i2):
                if origin_text[i] not in unk_tokens:
                    errors.append((origin_text[i], '', i))
                new_corrected_text += origin_text[i]
        elif tag == 'insert':
            for j in range(j1, j2):
                if corrected_text[j] not in unk_tokens:
                    errors.append(('', corrected_text[j], j))
                new_corrected_text += corrected_text[j]
        elif tag == 'equal':
            new_corrected_text += origin_text[i1:i2]

    errors = sorted(errors, key=lambda x: x[2])
    return new_corrected_text, errors


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
        print('--' * 42 + '\n')

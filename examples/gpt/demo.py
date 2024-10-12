# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append("../..")
from pycorrector.gpt.gpt_corrector import GptCorrector

if __name__ == '__main__':
    error_sentences = [
        '少先队员因该为老人让坐',
        '真麻烦你了。希望你们好好的跳无',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
        '目击者称，以军已把封锁路障除去，不过备关卡检查站仍有坦克派驻，车辆已能来往通过，只是流量很漫。',
        '行政相对人对行政机关作出的行政处罚决定不服,不能申请行政服役的是何种行政行为?',
        '“明德慎罚”是殷商初期“天命”、“天罚”思想的继承和发扬',
    ]
    m = GptCorrector("shibing624/chinese-text-correction-1.5b")

    batch_res = m.correct_batch(error_sentences, system_prompt="你是一个中文文本纠错助手。请根据用户提供的原始文本，生成纠正后的文本。")
    for i in batch_res:
        print(i)
        print()

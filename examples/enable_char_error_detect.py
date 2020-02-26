# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("../")
import pycorrector

if __name__ == '__main__':
    error_sentences = [
        '这纸厚度如何？质量怎么样',
        '双十一下单到现在还没发货的，',
        '一但工作效率提升，需要时间在工作上也减少',
        '可否送手机膜？送膜吗',
        '这水用来洗脸泡脚效果如何',
        '五香的不辣吧',
        '这款功率真有2000w吗',
        '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
        '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',  # 题高 => 提高 专业人氏 => 专业人士
        '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
        '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
    ]
    pycorrector.set_custom_confusion_dict(path='./my_confusion.txt')
    pycorrector.set_custom_word(path='./my_custom_word.txt')
    for line in error_sentences:
        correct_sent = pycorrector.correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

    print('*' * 53)

    pycorrector.enable_char_error(enable=False)
    # pycorrector.enable_word_error(enable=False)
    for line in error_sentences:
        # idx_errors = pycorrector.detect(line)
        # print(idx_errors)
        correct_sent = pycorrector.correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))

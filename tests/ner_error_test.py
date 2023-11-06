# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append("../")

from pycorrector import Corrector
ct = Corrector()


def test_disease():
    """测试疾病名纠错"""
    ct.enable_char_error(enable=False)
    error_sentence_1 = '这个新药奥美砂坦脂片能治疗心绞痛，效果还可以'  # 奥美沙坦酯片

    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '有个药名叫硫酸氢录吡各雷片能治疗高血压'  # 硫酸氢氯吡格雷片
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_brand():
    """测试品牌名纠错"""
    ct.enable_char_error(enable=False)
    error_sentence_1 = '买衣服就到拼哆哆'  # 拼多多
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '这个特仑素牛奶喝起来还不错吧'  # 特仑苏
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_person_name():
    """测试人名纠错"""
    error_sentence_1 = '发行人共同实际控制人萧华、霍荣铨、邓啟棠、张旗康分别'  # 误杀，萧华-肖
    import jieba.posseg
    print(jieba.posseg.lcut(error_sentence_1))
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '上述承诺内容系本人真实意思表示'  # 误杀：系-及
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_chengyu():
    """测试成语纠错"""
    ct.enable_char_error(enable=False)
    error_sentence_1 = '这块名表带带相传'  # 代代相传
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '他贰话不说把牛奶喝完了'  # 二话不说
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    # 这家伙还蛮格（（恪））尽职守的。
    # 报应接中迩（（而））来。
    # 人群穿（（川））流不息。
    # 这个消息不径（（胫））而走。
    # 眼前的场景美仑（（轮））美幻简直超出了人类的想象。
    # 看着这两个人谈笑风声（（生））我心理（（里））不由有些忌妒。
    # 有了这一番旁证（（征））博引。
    x = [
        '这场比赛我甘败下风',
        '这场比赛我甘拜下封',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',  # 接踵而来
        '人群穿流不息',
        '这个消息不径而走',
        '这个消息不胫儿走',
        '眼前的场景美仑美幻简直超出了人类的想象',
        '看着这两个人谈笑风声我心理不由有些忌妒',
        '有了这一番旁证博引',
        '有了这一番旁针博引',
    ]

    for i in x:
        print(i, ct.detect(i))
        print(i, ct.correct(i))

    ct.enable_char_error(enable=True)
    print("-" * 42)
    for i in x:
        print(i, ct.detect(i))
        print(i, ct.correct(i))


def test_suyu():
    """测试俗语纠错"""
    ct.enable_char_error(enable=False)

    error_sentence_1 = '这衣服买给她吧，也是肥水步流外人田'  # 肥水不流外人田
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))

    error_sentence_1 = '这么多字让他写也是赶鸭子打架'  # 赶鸭子上架
    correct_sent = ct.correct(error_sentence_1)
    print("original sentence:{} => correct sentence:{}".format(error_sentence_1, correct_sent))


def test_ner():
    from pycorrector.utils.tokenizer import segment
    from pycorrector import Corrector
    c = Corrector()
    c.check_corrector_initialized()
    c.check_detector_initialized()
    error_sentences = [
        '这个消息在北京城里不胫儿走',
        '大家已经满头大汉了，休息吧',
        '我不要你花钱,这些路曲近通幽',  # 曲径通幽
        '这个消息不胫儿走',
        '这个消息不径而走',  # 胫
        '真的是无稽之谈',
        '真的是无集之谈',  # 集
        '小丽宝儿的学习成绩一落千仗太失望了',
        '肉骨头是索然无味',
        '肉骨头是索染无味',  # 然
        '看书是一心一意，绝不东张夕望，好厉害。',  # 西
        "复方甘草口服液好喝吗",
        '新进人员时，知识当然还不过，可是人有很有精神，面对工作很认真的话，很快就学会、体会。',
    ]
    for line in error_sentences:
        print(line)
        print("segment:", segment(line))
        print("tokenize:", c.tokenizer.tokenize(line))
        print(c.detect(line))
        correct_sent = c.correct(line)
        print("original sentence:{} => correct sentence:{}".format(line, correct_sent))


def test_common_error():
    from pycorrector import Corrector
    from pycorrector.proper_corrector import load_dict_file
    m = Corrector()
    data = load_dict_file('./common_error_pairs.txt')
    error_sentences = ['我喜欢' + k for k, v in data.items()]
    error_sentences_val = [v for k, v in data.items()]
    for i, v in zip(error_sentences, error_sentences_val):
        print(i, ' -> ', m.correct(i))
        print(v)

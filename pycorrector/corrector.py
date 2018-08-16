# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: corrector with spell and stroke
import codecs
import os
import pdb
import time
import math

from collections import defaultdict
from pypinyin import lazy_pinyin

import jieba.posseg as pseg

import pycorrector.config as config
from pycorrector.detector import detect
from pycorrector.detector import get_frequency
from pycorrector.detector import get_ppl_score
from pycorrector.detector import trigram_char
from pycorrector.detector import word_freq
from pycorrector.utils.io_utils import dump_pkl
from pycorrector.utils.io_utils import get_logger
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.text_utils import is_chinese_string
from pycorrector.utils.text_utils import is_chinese
from pycorrector.utils.text_utils import traditional2simplified
from pycorrector.utils.text_utils import tokenize

pwd_path = os.path.abspath(os.path.dirname(__file__))
char_dict_path = os.path.join(pwd_path, config.char_dict_path)

default_logger = get_logger(__file__)


def load_char_dict(path):
    char_dict = ''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            char_dict += w.strip()
    return char_dict

def load_2char_dict(path):
    text = codecs.open(path, 'rb', encoding = 'utf-8').read()
    return set(text.split('\n'))

def load_word_dict(path):
    word_dict = set()
    word_dict_file = codecs.open(path, 'rb', encoding = 'utf-8').readlines()
    for line in word_dict_file:
        word_dict.add(line.split()[0])
    return word_dict

def load_same_pinyin(path, sep='\t'):
    """
    加载同音字
    :param path:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        default_logger.debug("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = traditional2simplified(line.strip())
            parts = line.split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                # same_pron_same_tone = set(list(parts[1]))
                # same_pron_diff_tone = set(list(parts[2]))
                # value = same_pron_same_tone.union(same_pron_diff_tone)
                value = set(list("".join(parts)))
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value

    # these pairs would be dealed with rule
    result['他'] -= {'她', '它'}
    result['她'] -= {'他', '它'}
    result['它'] -= {'她', '他'}
    result['影'] -= {'音'}

    return result

def load_same_stroke(path, sep=','):
    """
    加载形似字
    :param path:
    :param sep:
    :return:
    """
    result = defaultdict(set)
    if not os.path.exists(path):
        default_logger.debug("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = traditional2simplified(line.strip())
            parts = line.strip().split(sep)
            if parts and len(parts) > 1:
                for i, c in enumerate(parts):
                    # result[c].add(c)
                    # result[c] |= set(list(parts[:i] + parts[i + 1:]))
                    result[c] |= set(parts)
    return result

cn_char_set = load_char_dict(char_dict_path)
two_char_dict = load_2char_dict('../pycorrector/data/char_two_set.txt')

# # word dictionary
word_dict_text_path = os.path.join(pwd_path, config.word_dict_path)
word_dict_model_path = os.path.join(pwd_path, config.word_dict_model_path)
if os.path.exists(word_dict_model_path):
    cn_word_set = load_pkl(word_dict_model_path)
else:
    default_logger.debug('load word dict from text file:', word_dict_model_path)
    cn_word_set = load_word_dict(word_dict_text_path)
    dump_pkl(cn_word_set, word_dict_model_path)

# similar pronuciation
same_pinyin_text_path = os.path.join(pwd_path, config.same_pinyin_text_path)
same_pinyin_model_path = os.path.join(pwd_path, config.same_pinyin_model_path)
same_pinyin = load_same_pinyin(same_pinyin_text_path)
# if os.path.exists(same_pinyin_model_path):
#     same_pinyin = load_pkl(same_pinyin_model_path)
# else:
#     default_logger.debug('load same pinyin from text file:', same_pinyin_text_path)
#     same_pinyin = load_same_pinyin(same_pinyin_text_path)
#     dump_pkl(same_pinyin, same_pinyin_model_path)

# similar shape
same_stroke_text_path = os.path.join(pwd_path, config.same_stroke_text_path)
same_stroke_model_path = os.path.join(pwd_path, config.same_stroke_model_path)
if os.path.exists(same_stroke_model_path):
    same_stroke = load_pkl(same_stroke_model_path)
else:
    default_logger.debug('load same stroke from text file:', same_stroke_text_path)
    same_stroke = load_same_stroke(same_stroke_text_path)
    dump_pkl(same_stroke, same_stroke_model_path)

def get_same_pinyin(char):
    """
    取同音字
    :param char:
    :return:
    """
    return same_pinyin.get(char, set())


def get_same_stroke(char):
    """
    取形似字
    :param char:
    :return:
    """
    return same_stroke.get(char, set())


def edit_distance_word(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    return set(transposes + replaces)


def known(words):
    return set(word for word in words if word in word_freq)


def get_confusion_char_set(c):
    confusion_char_set = get_same_pinyin(c).union(get_same_stroke(c))
    if not confusion_char_set:
        confusion_char_set = {c}
    return confusion_char_set


def get_confusion_two_char_set(word):
    return set([char_1 + char_2 for char_1 in get_confusion_char_set(word[0]) \
                                for char_2 in get_confusion_char_set(word[1]) \
                                if char_1 + char_2 in cn_word_set])


def get_confusion_word_set(word):
    confusion_word_set = set()
    candidate_words = list(known(edit_distance_word(word, cn_char_set)))
    for candidate_word in candidate_words:
        if lazy_pinyin(candidate_word) == lazy_pinyin(word):
            # same pinyin
            confusion_word_set.add(candidate_word)
    return confusion_word_set


def _generate_items(sentence, idx, word, fraction=1):
    candidates_1_order = []
    candidates_2_order = []
    candidates_3_order = []

    # candidates_1_order.extend(get_confusion_word_set(word))

    if len(word) == 1:
        confusion = [i for i in get_confusion_char_set(word[0]) if i]
        candidates_2_order.extend(confusion)

    if len(word) > 1:

        def combine_two_confusion_char(sentence, idx, word):
            # # assuming there is only two char to change
            # # definitely not the final version, need to be fixed!!!!
            result = []
            for i in range(len(word) - 1):
                for j in range(i + 1,len(word)):
                    # for i_word in get_confusion_char_set(word[i]):
                    #     for j_word in get_confusion_char_set(word[j]):
                    #         if i == 0 
                    result.extend([word[: i] + i_word + word[i + 1: j] + j_word + word[j + 1:] \
                                   for i_word in get_confusion_char_set(word[i]) if i_word \
                                   for j_word in get_confusion_char_set(word[j]) if j_word])
            return result

        def confusion_set(sentence, idx, word):
            # maximum number of change char is set up by 'edit_distance'

            # the maximum edit-distance
            edit_distance = 2

            cands_tmp = [['',0]]
            result = set()
            ids = list(range(int(idx.split(',')[0]), int(idx.split(',')[1])))

            # # change individual char
            while cands_tmp:

                if len(cands_tmp[0][0]) == len(word):
                    result.add(cands_tmp[0][0])

                elif cands_tmp[0][1] == edit_distance:
                    result.add(cands_tmp[0][0] + word[len(cands_tmp[0][0]):])

                else:
                    target_idx = ids[len(cands_tmp[0][0])]
                    for char_cand in get_confusion_char_set(sentence[target_idx]):

                        if target_idx == 0:
                            if char_cand + sentence[target_idx + 1] not in two_char_dict:
                                continue

                        elif target_idx == len(sentence) - 1:
                            if sentence[target_idx - 1] + char_cand not in two_char_dict:
                                continue

                        elif char_cand + sentence[target_idx + 1] not in two_char_dict and \
                             sentence[target_idx - 1] + char_cand not in two_char_dict:
                            continue
                        
                        if char_cand == sentence[target_idx]:
                            cands_tmp.append([cands_tmp[0][0] + char_cand, cands_tmp[0][1]])
                        else:
                            cands_tmp.append([cands_tmp[0][0] + char_cand, cands_tmp[0][1] + 1])

                cands_tmp.pop(0)

            # # change connected two chars
            for i in range(len(word) - 1):
                for char_i in get_confusion_char_set(word[i]):
                    for char_j in get_confusion_char_set(word[i + 1]):
                        if char_i + char_j in two_char_dict:
                            result.add(word[:i] + char_i + char_j + word[i + 2:])



            return list(result)


        confusion = confusion_set(sentence, idx, word)
        # confusion  = combine_two_confusion_char(word)
        candidates_2_order.extend(confusion)


        # # same first pinyin
        # confusion = [i + word[1:] for i in get_confusion_char_set(word[0]) if i]
        # candidates_2_order.extend(confusion)

        # # same last pinyin
        # confusion = [word[:-1] + i for i in get_confusion_char_set(word[-1]) if i]
        # candidates_2_order.extend(confusion)


        # if len(word) > 2:
        #     # same mid char pinyin
        #     # for idx in range(1,len(word) - 1):
        #     #     confusion = [word[:idx] + i + word[idx + 1:] for i in get_confusion_char_set(word[idx])]
        #     confusion = [word[0] + i + word[2:] for i in get_confusion_char_set(word[1])]
        #     candidates_3_order.extend(confusion)

        #     # same first word pinyin
        #     confusion_word = [i + word[-1] for i in get_confusion_word_set(word[:-1])]
        #     candidates_1_order.extend(confusion_word)

        #     # same last word pinyin
        #     confusion_word = [word[0] + i for i in get_confusion_word_set(word[1:])]
        #     candidates_1_order.extend(confusion_word)

    # add all confusion word list
    confusion_word_set = set(candidates_1_order + candidates_2_order + candidates_3_order)
    confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
    confusion_sorted = sorted(confusion_word_list, key=lambda k: get_frequency(k), reverse=True)

    # #####################
    # print(len(confusion_word_set))
    # # print(confusion_sorted)
    # pdb.set_trace()
    # #####################

    return confusion_sorted[:len(confusion_word_list) // fraction + 1]


def get_sub_array(nums):
    """
    取所有连续子串，
    [0, 1, 2, 5, 7, 8]
    => [[0, 3], 5, [7, 9]]
    :param nums: sorted(list)
    :return:
    """
    ret = []
    for i, c in enumerate(nums):
        if i == 0:
            pass
        elif i <= ii:
            continue
        elif i == len(nums) - 1:
            ret.append([c])
            break
        ii = i
        cc = c
        # get continuity Substring
        while ii < len(nums) - 1 and nums[ii + 1] == cc + 1:
            ii = ii + 1
            cc = cc + 1
        if ii > i:
            ret.append([c, nums[ii] + 1])
        else:
            ret.append([c])
    return ret


def get_valid_sub_array(sentence, sub_array_list):
    """
    this function is to get rid of puctuation in detected string

    :param  sentence:    target sentence
            subarray:    index of suspected string
    :return valid_array: index of valid suspected string without punctuation
    """

    # print(sub_array_list)

    valid_array_detail = []

    for sub_array in sub_array_list:
        valid_sub_array_detail = []
        if len(sub_array) == 1:
            if is_chinese(sentence[sub_array[0]]):
                valid_array_detail.append([sub_array[0], sub_array[0]])
        else:
            for i in range(sub_array[0], sub_array[1]):
                if is_chinese(sentence[i]):
                    valid_sub_array_detail.append(i)
                elif valid_sub_array_detail:
                    valid_array_detail.append(valid_sub_array_detail)
                    valid_sub_array_detail = []
            if valid_sub_array_detail:
                valid_array_detail.append(valid_sub_array_detail)

    # print(valid_array_detail)
    return [[sub[0], sub[-1] + 1] for sub in valid_array_detail]


def count_diff(str1, str2):
    # # assuming len(str1) == len(str2)
    count = 0
    if len(str1) != len(str2):
        print(str1)
        print(str2)
        pdb.set_trace()
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count


def correct_stat(sentence, sub_sents):

    detail = []
    cands   = []

    for idx, item in sub_sents:

        maybe_error_items = _generate_items(sentence, idx, item)

        if not maybe_error_items:
            continue
        ids = idx.split(',')
        begin_id = int(ids[0])
        end_id = int(ids[-1]) if len(ids) > 1 else int(ids[0]) + 1
        before = sentence[:begin_id]
        after = sentence[end_id:]

        # ###################
        # print(item)
        # pdb.set_trace()
        # ###################
        factor1 = 4.5

        base_score = get_ppl_score(list(before + item + after), mode=trigram_char) \
                                + factor1 * count_diff(item, item)
        min_score  = base_score
        corrected_item = item
        for candidate in maybe_error_items:
            score = get_ppl_score(list(before + candidate + after), mode=trigram_char) \
                                + factor1 * count_diff(item, candidate)
            if score < min_score:
                corrected_item = candidate
                min_score = score

        delta_score = base_score - min_score
 
        cands.append([idx, corrected_item, delta_score])

    cands.sort(key = lambda x: x[2], reverse = True)

    factor2 = 9

    for i, [idx, corrected_item, delta_score] in enumerate(cands):
        if delta_score > i * factor2:
            idx = [int(idx.split(",")[0]), int(idx.split(",")[1])]
            detail.append(list(zip([sentence[idx[0]:idx[1]]], \
                                   [corrected_item],          \
                                   [idx[0]],                  \
                                   [idx[1]])))  
            
            sentence = sentence[: idx[0]] + \
                       corrected_item +     \
                       sentence[idx[1]:]
        else:
            break
    # ###################
    # print(detail)
    # pdb.set_trace()
    # ###################
    return sentence, detail


def get_sub_sent(idx, sentence):
    begin_id = 0
    end_id = 0
    for i in range(idx,-1,-1):
        if not is_chinese(sentence[i]):
            begin_id = i
            break
    for i in range(idx, len(sentence)):
        if not is_chinese(sentence[i]):
            end_id = i
            break
    return [begin_id, end_id]


def correct_rule(sentence, sub_sents):
    detail = []

    # # rule for '他她它'('he, she, it')
    # dict_hsi  = {
    #             '他' : {'爸','父','爷','哥','弟','兄','子','叔','伯','他','爹','先生'},
    #             '她' : {'妈','母','奶','姐','妹','姑姑','婶','姊','妯','娌','她','婆','姨','太太','夫人','娘'},
    #             '它' : {'它'}
    #             }
    # for i in range(len(sentence)):
    #     if sentence[i] in dict_hsi.keys():
    #         for key in dict_hsi.keys():
    #             if set(list(sentence[:i])) & dict_hsi[key]:
    #                 sentence = sentence[:i] + key + sentence[i + 1:]
    #                 detail.append([(sentence[i], key, i, i + 1)])
    #                 continue

    # # rule for '的地得'
    if set(sentence) & {'的', '地', '得'}:
        old_sentence = sentence

        seg = pseg.lcut(sentence)
        # # in the form of list of pair(w.word, w.flag)
        word = [w.word for w in seg]
        tag  = [w.flag for w in seg]

        for i in range(len(word)):
            if word[i] in {'的', '地', '得'} and 1 < i < len(word) - 1:
                # '地'
                if (tag[i + 1] == 'v' or \
                    word[i + 1] == '被' or \
                    tag[i + 1: i + 4] == ['p','n','v'] or \
                    tag[i + 1: i + 5] == ['p','n','f','v']) and \
                    (tag[i - 1] in {'i','d','ad','l'} or word[i-1] in {'一样','那么'}) and len(word[i - 1]) > 1 :
                    if i > 2 and tag[i - 2] in {'n','r','vn','an','d','x'}:
                        if word[i + 1] not in {'做法','看法','想法','行为','存在'}:
                            sentence = sentence[:len(''.join(word[:i]))] + \
                                       '地' +                              \
                                       sentence[len(''.join(word[:i])) + 1:]

                if tag[i + 1] == 'a' and \
                   (tag[i - 1] in {'d'} or word[i-1] in {'一样','那么'}) and \
                   (tag[i + 2] == 'x' or tuple(tag[i+2:i+4]) in {('y','x'),('ul','x')}):
                    sentence = sentence[:len(''.join(word[:i]))] + \
                               '地' +                              \
                               sentence[len(''.join(word[:i])) + 1:]

                if tag[i - 1] == 'd' and tag[i + 1] in {'r','a'} \
                   and i < len(word) - 2 and tag[i + 2] == 'v':
                    sentence = sentence[:len(''.join(word[:i]))] + \
                               '地' +                              \
                               sentence[len(''.join(word[:i])) + 1:]                    

                # '得'
                if tag[i - 1] == 'v' and tag[i + 1] in {'a','d'}:
                    if tag[i + 1] == 'a':
                        if i > len(word) - 5:
                            sentence = sentence[:len(''.join(word[:i]))] +   \
                                       '得' +                                \
                                       sentence[len(''.join(word[:i])) + 1:]
                        elif word[i + 2] not in {'的', '地', '得'} or         \
                             tag[i + 3] not in {'n','vn','r'}:
                            sentence = sentence[:len(''.join(word[:i]))] +   \
                                       '得' +                                \
                                       sentence[len(''.join(word[:i])) + 1:]
                    if tag[i + 1] == 'd':
                        sentence = sentence[:len(''.join(word[:i]))] +       \
                                   '得' +                                    \
                                   sentence[len(''.join(word[:i])) + 1:]
                if tag[i - 1] == 'a' and word[i + 1] == '多' and not is_chinese(word[i + 2]):
                    sentence = sentence[:len(''.join(word[:i]))] +       \
                               '得' +                                    \
                               sentence[len(''.join(word[:i])) + 1:]

                # for word '得到'
                if tag[i - 1] in {'n','r','vn'} and word[i + 1] == '到':
                    sentence = sentence[:len(''.join(word[:i]))] +       \
                               '得' +                                    \
                               sentence[len(''.join(word[:i])) + 1:]                                                            


                # '的'
                if tag[i - 1] == 'v' and (not is_chinese(word[i + 1]) or \
                                  (i < len(word) - 2 and not is_chinese(word[i + 2]) and tag[i + 1] == 'y')):
                    sentence = sentence[:len(''.join(word[:i]))] +       \
                               '的' +                                    \
                               sentence[len(''.join(word[:i])) + 1:] 

                if tag[i - 1] == 'n' and tag[i + 1] == 'n':
                    sentence = sentence[:len(''.join(word[:i]))] +       \
                               '的' +                                    \
                               sentence[len(''.join(word[:i])) + 1:]                    

            if word[i] in {'真得','真地'}:
                sentence = sentence[:len(''.join(word[:i]))] +        \
                           '真的' +                                    \
                           sentence[len(''.join(word[:i + 1])):]   

        for idx in range(len(sentence)):
            if sentence[idx] != old_sentence[idx]:
                detail.append([old_sentence[idx], sentence[idx], idx, idx + 1])

    # # rule for '啊阿'
    if set(sentence) & {'阿'}:
        old_sentence = sentence
        for i in range(len(sentence)):
            if sentence[i] == '阿' and not is_chinese(sentence[i + 1]):
                sentence = sentence[:i] + '啊' + sentence[i + 1:]

        for idx in range(len(sentence)):
            if sentence[idx] != old_sentence[idx]:
                detail.append([old_sentence[idx], sentence[idx], idx, idx + 1])

    # # 疑问代词：to suggest question
    ques_word = {'怎','什么','多少','谁', \
                 '可不可','是不是', '能不能','会不会'} # to be added
    # # 引导疑问句的词: to introduce a question
    intr_word = {'知道','想一想','无论','不管'} # to be added

    # # rule for '那哪'           // 目前还不能识别反问句
    if set(sentence) & {'那', '哪'}:
        old_sentence = sentence
        # for idx in detect(sentence):
        for idx in range(len(sentence)):
            if sentence[idx] in {'那', '哪'}:
                if idx < len(sentence) - 1 and sentence[idx + 1] == '么':
                    sentence = sentence[:idx] + '那' + sentence[idx + 1:]
                    continue
                [sub_sent_b, sub_sent_e] = get_sub_sent(idx, sentence)
                sub_sent = sentence[sub_sent_b: sub_sent_e + 1]

                # question sentence
                if sub_sent[-1] == '？':
                    if True in [i in sub_sent for i in ques_word]:
                        sentence = sentence[:idx] + '那' + sentence[idx + 1:]
                    else:
                        sentence = sentence[:idx] + '哪' + sentence[idx + 1:]

                # # state sentence
                else:
                    if True in [i in sub_sent[:idx - sub_sent_b] for i in intr_word]:
                        if True not in [i in sub_sent for i in ques_word]:
                            sentence = sentence[:idx] + '哪' + sentence[idx + 1:]
                    else:
                        sentence = sentence[:idx] + '那' + sentence[idx + 1:]

        for idx in range(len(sentence)):
            if sentence[idx] != old_sentence[idx]:
                detail.append([old_sentence[idx], sentence[idx], idx, idx + 1])


    return sentence, detail


def correct(sentence):

    detail = []

    maybe_error_ids = get_valid_sub_array(sentence, 
                                          get_sub_array(detect(sentence)))


    index_char_dict = dict()
    for index in maybe_error_ids:
        if len(index) == 1:

            index_char_dict[','.join(map(str, index))] = sentence[index[0]]
        else:

            index_char_dict[','.join(map(str, index))] = sentence[index[0]:index[-1]]


    sentence, detail_stat = correct_stat(sentence, index_char_dict.items())
    detail += detail_stat

    sentence, detail_rule = correct_rule(sentence, index_char_dict.items())
    detail += detail_rule


    return sentence, detail

# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


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


def get_sub_array(nums):
    """
    取所有连续子串，
    [0, 1, 2, 5, 7, 8]
    => [[0, 3], 5, [7, 9]]
    :param nums: sorted(list)
    :return:
    """
    ret = []
    ii = 0
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

import re

default_period = set(["。", "……", "！", "?", "？",  "\n",])
default_comma = set(["，", "，"])


def is_not_chinese_error(e):
    """不是全中文的情况， 忽略这类错误"""
    text = e[0]
    if len(text)==0:
        return True
    for char in text:
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
        if not chinese_char_pattern.match(char):
            return True
    return False


def long_sentence_split(text, max_length=128, period=None, comma=None):
    """
    先按照 period切分再按照 comma切分， 最后减少句子数量再合并
    """
    if period is None:
        period = default_period
    if comma is None:
        comma = default_comma
    
    def same_split(text, max_length=128):
        """
        等长切分
        """
        sentences = []
        for i in range(0, len(text), max_length):
            sentences.append(text[i:i + max_length])
        return sentences

    def get_sentences_by_punc(text, punc, max_length):
        n, last = len(text), 0
        sentences = []
        if n <= max_length:
            sentences.append(text)
        else:
            for i in range(n):
                if text[i] in punc:
                    sentences.extend(same_split(text[last:i + 1], max_length=max_length))
                    last = i + 1
            if last < n:
                sentences.extend(same_split(text[last:], max_length=max_length))
        return sentences
    
    sentences = []

    n, last = len(text), 0
    for i in range(n):
        if text[i] in period:
            sentences.extend(get_sentences_by_punc(text[last:i + 1], comma, max_length=max_length))
            last = i + 1
    if last < n:
        sentences.extend(get_sentences_by_punc(text[last:], comma, max_length=max_length))

    new = []
    cur = ""
    for s in sentences:
        if len(cur)+len(s)>max_length:
            new.append(cur)
            cur = ""
        cur += s
    if len(cur)>0:
        new.append(cur)
    return new
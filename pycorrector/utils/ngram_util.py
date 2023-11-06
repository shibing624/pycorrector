# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""


class NgramUtil:
    @staticmethod
    def unigrams(words):
        """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([words[i], words[i + k]]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in range(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for j in range(i + 1, L - 1):
                    for k in range(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
        Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in range(L - 3):
                for j in range(i + 1, L - 2):
                    for k in range(j + 1, L - 1):
                        for l in range(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        ngram = int(ngram)
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram
        elif ngram == 1234:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            fourgram = [x for x in NgramUtil.fourgrams(words, join_string) if len(x.split(join_string)) == 4]
            return unigram + bigram + trigram + fourgram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)

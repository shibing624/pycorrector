#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Translate chinese hanzi to pinyin by python
Created by Eric Lo on 2010-05-20.
Copyright (c) 2010 __lxneng@gmail.com__. http://lxneng.com All rights reserved.

Forked by skydarkchen <skydark2 at gmail>
"""

import os.path

try:
    chr = unichr
except NameError:
    pass

VERSION = '0.3a'


class Pinyin(object):
    """translate chinese hanzi to pinyin by python, inspired by flyerhzm’s
    `chinese\_pinyin`_ gem
    .. _chinese\_pinyin: https://github.com/flyerhzm/chinese_pinyin

    usage(python3)
    -----
    ::
        >>> p = Pinyin()
        >>> p.get_pinyin("上海")
        'shanghai'
        >>> p.get_pinyin("上海", tone=True)
        'shang4hai3'
        >>> p.get_initials("上")
        'S'
        >>> print(''.join(p.py2hz('shang4')))
        丄上姠尙尚蠰銄鑜
        >>> print(''.join(p.py2hz('a')))
        吖腌錒锕阿嗄阿阿啊阿
    """

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
            'Mandarin.dat')

    def __init__(self):
        self.dict = {}
        self.revdict = {}
        for line in open(self.data_path):
            k, v = line.strip().split('\t')
            v = v.lower().split(' ')
            hz = chr(int('0x%s' % k, 16))
            self.dict[hz] = v
            for vkey in v:
                self.revdict.setdefault(vkey, [])
                self.revdict[vkey].append(hz)

    def py2hz(self, pinyin):
        if pinyin == '':
            return []
        pinyin = pinyin.lower()
        if pinyin[-1].isdigit():
            return self.revdict.get(pinyin, [])
        ret = []
        for i in range(1, 6):
            key = '%s%s' % (pinyin, i)
            ret += self.revdict.get(key, [])
        return ret

    def get_pinyin(self, chars='', splitter='', tone=False):
        result = []
        for char in chars:
            v = self.dict.get(char, None)
            if v:
                v = v[0]
                if not tone and v[-1].isdigit():
                    v = v[:-1]
            else:
                v = char
            result.append(v)
        return splitter.join(result)

    def get_initials(self, char=''):
        if char == '':
            return ''
        return self.dict.get(char, [char])[0][0].upper()


if __name__ == '__main__':
    import unittest

    class PinyinTestCase(unittest.TestCase):
        def setUp(self):
            import sys
            py = sys.version_info
            self.py3k = py >= (3, 0, 0)

            self.py = Pinyin()

        def to_unicode(self, s):
            if self.py3k:
                return s
            return s.decode('utf-8')

        def test_get_pinyin(self):  ## test method names begin 'test*'
            s = self.to_unicode('上A2#海')
            a = self.to_unicode('shangA2#hai')
            aa = self.to_unicode('shang4A2#hai3')
            aaa = self.to_unicode('shang A 2 # hai')
            self.assertEqual(self.py.get_pinyin(s), a)
            self.assertEqual(self.py.get_pinyin(s, tone=True), aa)
            self.assertEqual(self.py.get_pinyin(s, splitter=' '), aaa)

        def test_get_initials(self):
            s = self.to_unicode('上')
            a = self.to_unicode('S')
            self.assertEqual(self.py.get_initials(s), a)

        def test_py2hz(self):
            s1 = self.to_unicode('shang4')
            s2 = self.to_unicode('a')
            a1 = self.to_unicode('丄上姠尙尚蠰銄鑜')
            a2 = self.to_unicode('吖腌錒锕阿嗄阿阿啊阿')
            self.assertEqual(''.join(self.py.py2hz(s1)), a1)
            self.assertEqual(''.join(self.py.py2hz(s2)), a2)

    unittest.main()

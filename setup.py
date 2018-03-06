# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from __future__ import print_function
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pycorrector',
    version='0.0.3',
    description='Chinese Text Error corrector',
    long_description=long_description,
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/corrector',
    license="MIT",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='NLP,correction,Chinese error corrector,corrector',
    install_requires=[
        'kenlm==0.0.0',
        'numpy>=1.13.1',
        'pypinyin==0.23.0'
    ],
    packages=['corrector'],
    package_dir={'corrector': 'corrector'},
    package_data={
        'corrector': ['*.py', 'zhtools/*', 'LICENSE', 'README.*', 'data/cn/*.txt'],
    }
)

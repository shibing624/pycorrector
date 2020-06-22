# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 
from __future__ import print_function

import sys

from setuptools import setup, find_packages

__version__ = None
exec(open('pycorrector/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for pycorrector.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='pycorrector',
    version=__version__,
    description='Chinese Text Error Corrector',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/pycorrector',
    license="Apache 2.0",
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='NLP,correction,Chinese error corrector,corrector',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'pycorrector': 'pycorrector'},
    package_data={'pycorrector': ['*.*', 'LICENSE', '../LICENSE', 'README.*', '../*.txt', 'data/*', 'data/en/big.txt',
                                  'utils/*.', 'bert/*', 'deep_context/*', 'conv_seq2seq/*', 'seq2seq_attention/*',
                                  'transformer/*', 'electra/*']}
)

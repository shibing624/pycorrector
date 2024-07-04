# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

from setuptools import setup, find_packages

__version__ = None
exec(open('pycorrector/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for pycorrector.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

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
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='pycorrector,correction,Chinese error correction,NLP',
    install_requires=[
        "jieba",
        "pypinyin",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "six",
        "loguru",
        "pyahocorasick",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'pycorrector': 'pycorrector'},
    package_data={'pycorrector': ['*.*', 'data/*', 'data/en.json.gz', 'data/sighan2015_test.tsv']}
)

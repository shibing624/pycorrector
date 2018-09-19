# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from __future__ import print_function
from setuptools import setup, find_packages
from pycorrector import __version__

long_description = '''
## Usage

### install
* pip3 install pycorrector 
* Or download https://github.com/shibing624/corrector Unzip and run python3 setup.py install

### correct 
input:
```
import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

```

output:
```
少先队员应该为老人让座 [[('因该', '应该', 4, 6)], [('坐', '座', 10, 11)]]
```

----


# corrector
中文错别字纠正工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python开发。

**corrector**依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。

## 特征
### 语言模型
* Kenlm（统计语言模型工具）
* RNNLM（TensorFlow、PaddlePaddle均有实现栈式双向LSTM的语言模型）

## 使用说明

### 安装
* 全自动安装：pip3 install pycorrector 
* 半自动安装：下载 https://github.com/shibing624/corrector 解压缩并运行 python3 setup.py install

### 纠错 
使用示例:
```
import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

```

输出:
```
少先队员应该为老人让座 [[('因该', '应该', 4, 6)], [('坐', '座', 10, 11)]]
```  
    
'''

setup(
    name='pycorrector',
    version=__version__,
    description='Chinese Text Error corrector',
    long_description=long_description,
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/corrector',
    license="Apache 2.0",
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='NLP,correction,Chinese error corrector,corrector',
    install_requires=[
        'scipy',
        'scikit-learn',
        'pypinyin',
        'kenlm',
        'jieba',
        'tensorflow',
        'keras>=2.1.5',
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'pycorrector': 'pycorrector'},
    package_data={
        'pycorrector': ['*.*', 'LICENSE', 'README.*', 'data/*', 'data/kenlm/*', 'utils/*'],
    }
)

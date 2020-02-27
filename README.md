![alt text](docs/logo.svg)

[![PyPI version](https://badge.fury.io/py/pycorrector.svg)](https://badge.fury.io/py/pycorrector)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/pycorrector/LICENSE)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Python3](https://img.shields.io/badge/Python-3.X-red.svg)


# pycorrector

中文文本纠错工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python3开发。

**pycorrector**依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。

## Demo

https://www.borntowin.cn/product/corrector

## Question

中文文本纠错任务，常见错误类型包括：

- 谐音字词，如 配副眼睛-配副眼镜
- 混淆音字词，如 流浪织女-牛郎织女
- 字词顺序颠倒，如 伍迪艾伦-艾伦伍迪
- 字词补全，如 爱有天意-假如爱有天意
- 形似字错误，如 高梁-高粱
- 中文拼音全拼，如 xingfu-幸福
- 中文拼音缩写，如 sz-深圳
- 语法错误，如 想象难以-难以想象

当然，针对不同业务场景，这些问题并不一定全部存在，比如输入法中需要处理前四种，搜索引擎需要处理所有类型，语音识别后文本纠错只需要处理前两种，
其中'形似字错误'主要针对五笔或者笔画手写输入等。


## Solution
### 规则的解决思路
1. 中文纠错分为两步走，第一步是错误检测，第二步是错误纠正；
2. 错误检测部分先通过结巴中文分词器切词，由于句子中含有错别字，所以切词结果往往会有切分错误的情况，这样从字粒度和词粒度两方面检测错误，
整合这两种粒度的疑似错误结果，形成疑似错误位置候选集；
3. 错误纠正部分，是遍历所有的疑似错误位置，并使用音似、形似词典替换错误位置的词，然后通过语言模型计算句子困惑度，对所有候选集结果比较并排序，得到最优纠正词。

### 深度模型的解决思路
1. 端到端的深度模型可以避免人工提取特征，减少人工工作量，RNN序列模型对文本任务拟合能力强，rnn_attention在英文文本纠错比赛中取得第一名成绩，证明应用效果不错；
2. CRF会计算全局最优输出节点的条件概率，对句子中特定错误类型的检测，会根据整句话判定该错误，阿里参赛2016中文语法纠错任务并取得第一名，证明应用效果不错；
3. seq2seq模型是使用encoder-decoder结构解决序列转换问题，目前在序列转换任务中（如机器翻译、对话生成、文本摘要、图像描述）使用最广泛、效果最好的模型之一。


## Feature
### 模型
* kenlm：kenlm统计语言模型工具
* rnn_attention模型：参考Stanford University的nlc模型，该模型是参加2014英文文本纠错比赛并取得第一名的方法
* rnn_crf模型：参考阿里巴巴2016参赛中文语法纠错比赛CGED2018并取得第一名的方法(整理中)
* seq2seq_attention模型：在seq2seq模型加上attention机制，对于长文本效果更好，模型更容易收敛，但容易过拟合
* transformer模型：全attention的结构代替了lstm用于解决sequence to sequence问题，语义特征提取效果更好
* bert模型：中文fine-tuned模型，使用MASK特征纠正错字
* conv_seq2seq模型：基于Facebook出品的fairseq，北京语言大学团队改进ConvS2S模型用于中文纠错，在NLPCC-2018的中文语法纠错比赛中，是唯一使用单模型并取得第三名的成绩


### 错误检测
* 字粒度：语言模型困惑度（ppl）检测某字的似然概率值低于句子文本平均值，则判定该字是疑似错别字的概率大。
* 词粒度：切词后不在词典中的词是疑似错词的概率大。


### 错误纠正
* 通过错误检测定位所有疑似错误后，取所有疑似错字的音似、形似候选词，
* 使用候选词替换，基于语言模型得到类似翻译模型的候选排序结果，得到最优纠正词。


### 思考
1. 现在的处理手段，在词粒度的错误召回还不错，但错误纠正的准确率还有待提高，更多优质的纠错集及纠错词库会有提升，我更希望算法上有更大的突破。
2. 另外，现在的文本错误不再局限于字词粒度上的拼写错误，需要提高中文语法错误检测（CGED, Chinese Grammar Error Diagnosis）及纠正能力，列在TODO中，后续调研。



## Install
* 全自动安装：pip3 install pycorrector
* 半自动安装：
```
git clone https://github.com/shibing624/pycorrector.git
cd pycorrector
python3 setup.py install
```

通过以上两种方法的任何一种完成安装都可以。如果不想安装，可以下载[github源码包](https://github.com/shibing624/pycorrector/archive/master.zip)，安装下面依赖再使用。

#### 安装依赖
```
pip3 install -r requirements.txt
```

## Usage

- 文本纠错

```
import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

```

output:
```
少先队员应该为老人让座 [[('因该', '应该', 4, 6)], [('坐', '座', 10, 11)]]
```

> 规则方法默认会从路径`~/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm`加载kenlm语言模型文件，如果检测没有该文件，则程序会自动联网下载。当然也可以手动下载[模型文件(2.8G)](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)并放置于该位置。


- 错误检测
```

import pycorrector

idx_errors = pycorrector.detect('少先队员因该为老人让坐')
print(idx_errors)

```

output:
```
[['因该', 4, 6, 'word'], ['坐', 10, 11, 'char']]
```
> 返回类型是`list`, `[error_word, begin_pos, end_pos, error_type]`，`pos`索引位置以0开始。


- 关闭字粒度纠错

默认字粒度、词粒度的纠错都打开，一般情况下单字错误发生较少，而且字粒度纠错准确率较低。关闭字粒度纠错，这样可以提高纠错准确率，提高纠错速度。
```

import pycorrector

error_sentence_1 = '我的喉咙发炎了要买点阿莫细林吃'
pycorrector.enable_char_error(enable=False)
correct_sent = pycorrector.correct(error_sentence_1)
print(correct_sent)

```

output:
```
'我的喉咙发炎了要买点阿莫西林吃', [['细林', '西林', 12, 14]]
```
> 默认`enable_char_error`方法的`enable`参数为`True`，即打开错字纠正，这种方式纠错召回高一些，但是整体准确率会低；

> 如果追求准确率而不追求召回率的话，建议将`enable`设为`False`，仅使用错词纠正。


- 加载自定义混淆集

通过加载自定义混淆集，支持用户纠正已知的错误。

```
import pycorrector

pycorrector.set_log_level('INFO')

error_sentence_1 = '买iPhone差，要多少钱'
correct_sent = pycorrector.correct(error_sentence_1)
print(correct_sent)

print('*' * 53)

pycorrector.set_custom_confusion_dict(path='./my_custom_confusion.txt')
correct_sent = pycorrector.correct(error_sentence_1)
print(correct_sent)

```

output:
```
('买iPhone差，要多少钱', [])
*****************************************************
('买iphoneX，要多少钱', [['iphone差', 'iphoneX', 1, 8]])
```

具体demo见[example/use_custom_confusion.py](./examples/use_custom_confusion.py)，其中`./my_custom_confusion.txt`的内容格式如下，以空格间隔：
```
iphone差 iphoneX 100
```
> `set_custom_confusion_dict`方法的`path`参数为用户自定义混淆集文件路径。


- 加载自定义语言模型

默认提供下载并使用的kenlm语言模型`zh_giga.no_cna_cmn.prune01244.klm`文件是2.8G，内存较小的电脑使用`pycorrector`程序可能会吃力些。

支持用户加载自己训练的kenlm语言模型，或使用2014版人民日报数据训练的[模型](https://www.borntowin.cn/mm/emb_models/people_chars_lm.klm)，模型小（20M），准确率低些。

```
from pycorrector import Corrector

pwd_path = os.path.abspath(os.path.dirname(__file__))
lm_path = os.path.join(pwd_path, './people_chars_lm.klm')
model = Corrector(language_model_path=lm_path)

corrected_sent, detail = model.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)

```

output:
```
少先队员应该为老人让座 [[('因该', '应该', 4, 6)], [('坐', '座', 10, 11)]]
```

具体demo见[example/load_custom_language_model.py](./examples/load_custom_language_model.py)，其中`./people_chars_lm.klm`是自定义语言模型文件。

### Evaluate

提供评估脚本[pycorrector/utils/eval.py](./pycorrector/utils/eval.py)，该脚本有两个功能：
- 构建评估样本集：自动生成评估集[pycorrector/data/eval_corpus.json](pycorrector/data/eval_corpus.json), 包括字粒度错误100条、词粒度错误100条、语法错误100条，正确句子200条。用户可以修改条数生成其他评估样本分布。
- 计算纠错准召率：采用保守计算方式，简单把纠错之后与正确句子完成匹配的视为正确，否则为错。

执行该脚本后得到，规则方法纠错效果评估如下：
```
- 准确率：320/500=64%
- 召回率：152/300=50.67%
```
看来还有比较大的提升空间，误杀和漏召回的都有。

## 深度模型使用说明

### 安装依赖
```
pip3 install -r requirements-dev.txt
```

### 介绍

本项目的初衷之一是比对、共享各种文本纠错方法，抛砖引玉的作用，如果对大家在文本纠错任务上有一点小小的启发就是我莫大的荣幸了。

主要使用了多种深度模型应用于文本纠错任务，分别是前面`模型`小节介绍的`conv_seq2seq`、`seq2seq_attention`、
`transformer`、`bert`，各模型方法内置于`pycorrector`文件夹下，有`README.md`详细指导，各模型可独立运行，相互之间无依赖。


### 使用方法
各模型均可独立的预处理数据、训练、预测，下面以其中`seq2seq_attention`为例：

seq2seq_attention 模型使用示例:

#### 配置

通过修改`config.py`。


#### 数据预处理
```
cd seq2seq_attention
# 数据预处理
python preprocess.py

```
自动新建文件夹output，在output下生成`train.txt`和`test.txt`文件，以TAB（"\t"）间隔错误文本和纠正文本，文本以空格切分词，文件内容示例：

```
希 望 少 吸 烟 。	 希 望 烟 民 们 少 吸 烟 。
以 前 ， 包 括 中 国 ， 我 国 也 是 。	以 前 ， 不 仅 中 国 ， 我 国 也 是 。
我 现 在 好 得 多 了 。	我 现 在 好 多 了 。
```


#### 训练
```
python train.py
```
训练过程截图：
![train image](./docs/git_image/seq2seq_train.png)


#### 预测
```
python infer.py
```


预测输出效果样例：
```
input: 少先队员因该给老人让坐 output: 少先队员因该给老人让座
input: 少先队员应该给老人让坐 output: 少先队员应该给老人让座
input: 没有解决这个问题， output: 没有解决这个问题，，
input: 由我起开始做。 output: 由我起开始做
input: 由我起开始做 output: 由我开始做

```


## 自定义语言模型

语言模型对于纠错步骤至关重要，当前默认使用的是从千兆中文文本训练的中文语言模型[zh_giga.no_cna_cmn.prune01244.klm(2.8G)](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)。

大家可以用中文维基（繁体转简体，pycorrector.utils下有此功能）等语料数据训练通用的语言模型，或者也可以用专业领域语料训练更专用的语言模型。更适用的语言模型，对于纠错效果会有比较好的提升。


1. kenlm语言模型训练工具的使用，请见博客：http://blog.csdn.net/mingzai624/article/details/79560063
2. 附上训练语料<人民日报2014版熟语料>，包括：
    1）标准人工切词及词性数据people2014.tar.gz，
    2）未切词文本数据people2014_words.txt，
    3）kenlm训练字粒度语言模型文件及其二进制文件people2014corpus_chars.arps/klm，
    4）kenlm词粒度语言模型文件及其二进制文件people2014corpus_words.arps/klm。

人民日报2014版熟语料，网盘链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11。尊重版权，传播请注明出处。


## 中文纠错数据集
1. NLPCC 2018 GEC官方数据集[NLPCC2018-GEC](http://tcci.ccf.org.cn/conference/2018/taskdata.php)，
训练集[trainingdata](http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz)[解压后114.5MB]，该数据格式是原始文本，未做切词处理。
2. 汉语水平考试（HSK）和lang8原始平行语料[HSK+Lang8](https://pan.baidu.com/s/18JXm1KGmRu3Pe45jt2sYBQ)[190MB]，该数据集已经切词，可用作数据扩增
3. 以上语料，再加上CGED16、CGED17、CGED18的数据，经过以字切分，繁体转简体，打乱数据顺序的预处理后，生成用于纠错的熟语料(nlpcc2018+hsk)，网盘链接:https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130万对句子，215MB]

## 贡献及优化点

- [x] 优化形似字字典，提高形似字纠错准确率
- [x] 整理中文纠错训练数据，使用seq2seq做深度中文纠错模型
- [x] 添加中文语法错误检测及纠正能力
- [x] 规则方法添加用户自定义纠错集，并将其纠错优先度调为最高
- [x] seq2seq_attention 添加dropout，减少过拟合
- [x] 在seq2seq模型框架上，新增Pointer-generator network、Beam search、Unknown words replacement、Coverage mechanism等特性
- [x] 更新bert的fine-tuned使用wiki，适配transformers 2.2.1库
- [x] 升级代码，兼容TensorFlow 2.0库
- [ ] 支持人名、地名、机构名、药品名等自定义保护词典的文本纠错
- [ ] 升级bert纠错逻辑，提升基于mask的纠错效率

## 讨论群

微信交流群，感兴趣的同学可以加入沟通NLP文本纠错相关技术，issues上回复不及时也可以在群里面提问。

PS: 由于微信群满100人了，扫码加不了。加我*微信号：xuming624, 备注：个人名称-NLP纠错* 进群。

## 参考

* [基于文法模型的中文纠错系统](https://blog.csdn.net/mingzai624/article/details/82390382)
* [Norvig’s spelling corrector](http://norvig.com/spell-correct.html)
* [Chinese Spelling Error Detection and Correction Based on Language Model, Pronunciation, and Shape[Yu, 2013]](http://www.aclweb.org/anthology/W/W14/W14-6835.pdf)
* [Chinese Spelling Checker Based on Statistical Machine Translation[Chiu, 2013]](http://www.aclweb.org/anthology/O/O13/O13-1005.pdf)
* [Chinese Word Spelling Correction Based on Rule Induction[yeh, 2014]](http://aclweb.org/anthology/W14-6822)
* [Neural Language Correction with Character-Based Attention[Ziang Xie, 2016]](https://arxiv.org/pdf/1603.09727.pdf)
* [Chinese Spelling Check System Based on Tri-gram Model[Qiang Huang, 2014]](http://www.anthology.aclweb.org/W/W14/W14-6827.pdf)
* [Neural Abstractive Text Summarization with Sequence-to-Sequence Models[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)
* [基于深度学习的中文文本自动校对研究与实现[杨宗霖, 2019]](https://github.com/shibing624/pycorrector/blob/master/docs/基于深度学习的中文文本自动校对研究与实现.pdf)
* [A Sequence to Sequence Learning for Chinese Grammatical Error Correction[Hongkai Ren, 2018]](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_36)



----

# pycorrector
Chinese text error correction tool. 


**pycorrector** Use the language model to detect errors, pinyin feature and shape feature to correct chinese text 
error, it can be used for Chinese Pinyin and stroke input method.

## Features
### language model
* Kenlm
* RNNLM

## Usage

### install
* pip install pycorrector / pip3 install pycorrector 
* Or download https://github.com/shibing624/pycorrector, Unzip and run: python setup.py install

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


### Future work
1. P(c), the language model. We could create a better language model by collecting more data, and perhaps by using a 
little English morphology (such as adding "ility" or "able" to the end of a word).

2. P(w|c), the error model. So far, the error model has been trivial: the smaller the edit distance, the smaller the 
error.
Clearly we could use a better model of the cost of edits. get a corpus of spelling errors, and count how likely it is
to make each insertion, deletion, or alteration, given the surrounding characters. 

3. It turns out that in many cases it is difficult to make a decision based only on a single word. This is most 
obvious when there is a word that appears in the dictionary, but the test set says it should be corrected to another 
word anyway:
correction('where') => 'where' (123); expected 'were' (452)
We can't possibly know that correction('where') should be 'were' in at least one case, but should remain 'where' in 
other cases. But if the query had been correction('They where going') then it seems likely that "where" should be 
corrected to "were".

4. Finally, we could improve the implementation by making it much faster, without changing the results. We could 
re-implement in a compiled language rather than an interpreted one. We could cache the results of computations so 
that we don't have to repeat them multiple times. 
One word of advice: before attempting any speed optimizations, profile carefully to see where the time is actually 
going.


### Further Reading
* [Roger Mitton has a survey article on spell checking.](http://www.dcs.bbk.ac.uk/~roger/spellchecking.html)

# Reference
1. [Norvig’s spelling corrector](http://norvig.com/spell-correct.html)
2. [Norvig’s spelling corrector(java version)](http://raelcunha.com/spell-correct/)


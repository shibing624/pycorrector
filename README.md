# pycorrector
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg)

中文错别字纠正工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python3开发。

**pycorrector**依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。

## 特征
### 语言模型
* Kenlm（统计语言模型工具）
* RNNLM（TensorFlow、PaddlePaddle均有实现栈式双向LSTM的语言模型）

### 错误检测
* 字粒度：语言模型困惑度（ppl）检测某字的似然概率值低于句子文本平均值，则判定该字是疑似错别字的概率大。
* 词粒度：切词后不在词典中的词是疑似错词的概率大。


### 错误纠正
1. 通过错误检测定位所有疑似错误后，取所有疑似错字的音似、形似候选词，
2. 使用候选词替换，基于语言模型得到类似翻译模型的候选排序结果，得到最优纠正词。


### 思考
1. 现在的处理手段，在词粒度的错误召回还不错，但错误纠正的准确率还有待提高，更多优质的纠错集及纠错词库会有提升，我更希望算法上有更大的突破。
2. 另外，现在的文本错误不再局限于字词粒度上的拼写错误，需要提高中文语法错误检测（CGED, Chinese Grammar Error Diagnosis）及纠正能力，列在TODO中，后续调研。

## demo

http://www.borntowin.cn/nlp/corrector.html


## 使用说明

### 依赖
pip3 install -r requirements.txt

pip3 install git+https://www.github.com/keras-team/keras-contrib.git


### 安装
* 全自动安装：pip3 install pycorrector
* 半自动安装：
```
git clone https://github.com/shibing624/pycorrector.git
cd pycorrector
python3 setup.py install
```


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

## 自定义语言模型

语言模型对于纠错步骤至关重要，目前我能收集到的语料数据有人民日报数据。大家可以用中文维基（繁体转简体，pycorrector.utils下有此功能）等更大的语料数据训练效果更好的语言模型，
对于纠错效果会有比较好的提升。

1. kenlm语言模型训练工具的使用，请见博客：http://blog.csdn.net/mingzai624/article/details/79560063
2. 附上训练语料<人民日报2014版熟语料>，包括：
    1）标准人工切词及词性数据people2014.tar.gz，
    2）未切词文本数据people2014_words.txt，
    3）kenlm训练字粒度语言模型文件及其二进制文件people2014corpus_chars.arps/klm，
    4）kenlm词粒度语言模型文件及其二进制文件people2014corpus_words.arps/klm。

网盘链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11。尊重版权，传播请注明出处。

## 贡献及优化点

- [x] 使用RNN语言模型来提高纠错准确率。
- [x] 优化形似字字典，提高形似字纠错准确率。
- [x] 整理中文纠错集，使用seq2seq做深度中文纠错模型。
- [x] 添加中文语法错误检测及纠正能力。

## 参考

1. [基于文法模型的中文纠错系统](https://blog.csdn.net/mingzai624/article/details/82390382)
2.  [Norvig’s spelling corrector](http://norvig.com/spell-correct.html)
3. [《Chinese Spelling Error Detection and Correction Based on Language Model, Pronunciation, and Shape》[Yu, 2013]](http://www.aclweb.org/anthology/W/W14/W14-6835.pdf)
4. [《Chinese Spelling Checker Based on Statistical Machine Translation》[Chiu, 2013]](http://www.aclweb.org/anthology/O/O13/O13-1005.pdf)
5. [《Chinese Word Spelling Correction Based on Rule Induction》[yeh, 2014]](http://aclweb.org/anthology/W14-6822)
6. [《Neural Language Correction with Character-Based Attention》[Ziang Xie, 2016]](https://arxiv.org/pdf/1603.09727.pdf)


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


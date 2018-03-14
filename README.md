# corrector
Chinese text error correction tool. 


**corrector** Use the language model to detect errors, pinyin feature and shape feature to correct chinese text 
error, it can be used for Chinese Pinyin and stroke input method.

## Features
### language model
* Kenlm
* RNNLM

## Usage

### install
* pip install pycorrector / pip3 install pycorrector 
* Or download https://github.com/shibing624/corrector, Unzip and run python setup.py install

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
* 全自动安装：pip install pycorrector 或者 pip3 install pycorrector 
* 半自动安装：下载 https://github.com/shibing624/corrector, 解压缩并运行 python setup.py install

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

语言模型对于纠错步骤至关重要，目前我能收集到的语料数据只有人民日报数据。大家可以用中文维基（繁体转简体）等更大的语料数据训练效果更好的语言模型，
对于纠错效果会有很好的提升。

1. kenlm工具使用见博客：http://blog.csdn.net/mingzai624/article/details/79560063
2. 附上训练语料<人民日报2014版熟语料>，包括：
1）标准人工切词及词性数据people2014.tar.gz，
2）未切词文本数据people2014_words.txt，
3）kenlm训练字粒度语言模型文件及其二进制文件people2014corpus_chars.arps/klm，
4）kenlm词粒度语言模型文件及其二进制文件people2014corpus_words.arps/klm。
网盘地址：链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11。尊重版权，传播请注明出处。

## 贡献及优化点

* 升级使用RNN语言模型来评估纠错效果。
* 优化形似字字典，提高形似字纠错准确率。
* 整理中文纠错集，使用seq2seq做深度中文纠错模型
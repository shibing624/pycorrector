# corrector
Chinese spelling correction tool. 


**corrector**Use the language model to detect errors, phonetic features and resemblance characteristics to correcting 
error, it can be used for Chinese Pinyin and stroke input method.

## Features
### language model
* Kenlm
* RNNLM

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

====

# corrector
中文错别字纠正工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python开发。

**corrector**依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。

## Features
### 语言模型
* Kenlm（统计语言模型工具）
* RNNLM（TensorFlow、PaddlePaddle均有实现栈式双向LSTM的语言模型）

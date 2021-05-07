English | [简体中文](README.md)

![alt text](docs/pycorrector.png)

[![PyPI version](https://badge.fury.io/py/pycorrector.svg)](https://badge.fury.io/py/pycorrector)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/pycorrector/LICENSE)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Python3](https://img.shields.io/badge/Python-3.X-red.svg)


# pycorrector

Chinese Text Error Correction Tool.


**pycorrector** Use the language model to detect errors, pinyin feature and shape feature to correct chinese text
error, it can be used for Chinese Pinyin and stroke input method.

## Features
### language model
* Kenlm
* RNNLM

### deep model
* rnn_attention
* seq2seq_attention
* conv_seq2seq
* transformer
* bert
* electra


## Install
* auto：pip install pycorrector
* manual：
```
git clone https://github.com/shibing624/pycorrector.git
cd pycorrector
python setup.py install
```


#### Install Requires
* install kenlm
```
pip install https://github.com/kpu/kenlm/archive/master.zip
```

* install others
```
pip install -r requirements.txt
```

## Usage

- Text Correction

```python
import pycorrector

corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)
```

output:
```
少先队员应该为老人让座 [[('因该', '应该', 4, 6)], [('坐', '座', 10, 11)]]
```

> model load from: `~/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm`, if not download auto, do it from [file(2.8G)](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm).Correction



- Error Detection
```python
import pycorrector

idx_errors = pycorrector.detect('少先队员因该为老人让坐')
print(idx_errors)
```

output:
```
[['因该', 4, 6, 'word'], ['坐', 10, 11, 'char']]
```
> return `list`, `[error_word, begin_pos, end_pos, error_type]`，`pos` index starts with 0.


- English Seplling Error Correction


```python
import pycorrector

sent_lst = ['what', 'hapenning', 'how', 'to', 'speling', 'it', 'you', 'can', 'gorrect', 'it']
for i in sent_lst:
    print(i, '=>', pycorrector.en_correct(i))
```

output:
```
what => what
hapenning => happening
how => how
to => to
speling => spelling
it => it
you => you
can => can
gorrect => correct
it => it
```


### Command Line Usage
- Command line

```
python -m pycorrector -h
usage: __main__.py [-h] -o OUTPUT [-n] [-d] input

@description:

positional arguments:
  input                 the input file path, file encode need utf-8.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the output file path.
  -n, --no_char         disable char detect mode.
  -d, --detail          print detail info

```

case：
```
python -m pycorrector input.txt -o out.txt -n -d
```
> input file：`input.txt`; output file：`out.txt `


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


## Cite

```latex
@software{pycorrector,
  author = {Xu Ming},
  title = {{pycorrector: Text Error Correction Tool}},
  year = {2020},
  url = {https://github.com/shibing624/pycorrector},
}
```

## License

**Apache License 2.0**

## References

* [Norvig’s spelling corrector](http://norvig.com/spell-correct.html)
* [Chinese Spelling Error Detection and Correction Based on Language Model, Pronunciation, and Shape[Yu, 2013]](http://www.aclweb.org/anthology/W/W14/W14-6835.pdf)
* [Chinese Spelling Checker Based on Statistical Machine Translation[Chiu, 2013]](http://www.aclweb.org/anthology/O/O13/O13-1005.pdf)
* [Chinese Word Spelling Correction Based on Rule Induction[yeh, 2014]](http://aclweb.org/anthology/W14-6822)
* [Neural Language Correction with Character-Based Attention[Ziang Xie, 2016]](https://arxiv.org/pdf/1603.09727.pdf)
* [Chinese Spelling Check System Based on Tri-gram Model[Qiang Huang, 2014]](http://www.anthology.aclweb.org/W/W14/W14-6827.pdf)
* [Neural Abstractive Text Summarization with Sequence-to-Sequence Models[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)
* [A Sequence to Sequence Learning for Chinese Grammatical Error Correction[Hongkai Ren, 2018]](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_36)
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)

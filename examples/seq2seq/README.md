# ConvSeq2seq Attention Model for Chinese Spelling Correction


## Features

* 基于Attention机制的Sequence to Sequence模型
* Luong Attention
* Conv Seq2Seq model, GPU并行计算，训练加速
* 训练加速tricks：dataset bucketing, prefetching, token-based batching, gradients accumulation
* Beam Search
* Chinese Samples: sighan2015 data

## Usage

### Requirements
* pip安装依赖包
```
torch>=1.4.0
transformers>=4.4.2
```


### 快速加载
#### pycorrector快速预测

example: [examples/seq2seq/demo.py](https://github.com/shibing624/pycorrector/blob/master/examples/seq2seq/demo.py)
```python
from pycorrector import ConvSeq2SeqCorrector
m = ConvSeq2SeqCorrector()
print(m.correct_batch(['今天新情很好', '你找到你最喜欢的工作，我也很高心。']))
```

output:
```shell
[{'source': '今天新情很好', 'target': '今天心情很好', 'errors': [('新', '心', 2)]},
{'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}]
```

### Dataset

#### toy data
sighan 2015中文拼写纠错数据（2k条）：[examples/data/sighan_2015/train.tsv](https://github.com/shibing624/pycorrector/blob/master/examples/data/sighan_2015/train.tsv)

data format:
```
你说的是对，跟那些失业的人比起来你也算是辛运的。	你说的是对，跟那些失业的人比起来你也算是幸运的。
```


#### big train data

nlpcc2018+hsk dataset, download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB] 

### Train model
run train:
```
python train.py --do_train --do_predict
```


### Predict model
```
python predict.py
```

output:
```shell
[{'source': '今天新情很好', 'target': '今天心情很好', 'errors': [('新', '心', 2)]},
{'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}]
```
![result image](https://github.com/shibing624/pycorrector/blob/master/docs/git_image/convseq2seq_ret.png)


## Release model
基于SIGHAN2015数据集训练的convseq2seq模型，已经release到github:

- convseq2seq model: https://github.com/shibing624/pycorrector/releases/download/0.4.5/convseq2seq_correction.tar.gz

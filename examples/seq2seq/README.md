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

### Dataset

#### toy data
sighan 2015中文拼写纠错数据（2k条）：[examples/data/sighan_2015/train.tsv](https://github.com/shibing624/pycorrector/blob/master/examples/data/sighan_2015/train.tsv)

data format:
```
# head -n 1 train.txt
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
```
input  : 老是较书。
predict: 老师教书。

input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
predict: 感谢等五分以后，碰到一位很棒的女生跟我可聊。

input  : 遇到一位很棒的奴生跟我聊天。
predict: 遇到一位很棒的女生跟我聊天。

input  : 遇到一位很美的女生跟我疗天。
predict: 遇到一位很美的女生跟我疗天。

input  : 他们只能有两个选择：接受降新或自动离职。
predict: 他们只能有两个选择：接受降薪或自动离职。

input  : 王天华开心得一直说话。
predict: 王天华开心地一直说话。

```
![result image](https://github.com/shibing624/pycorrector/blob/master/docs/git_image/convseq2seq_ret.png)


## Release model
基于SIGHAN2015数据集训练的convseq2seq模型，已经release到github:

- convseq2seq model: https://github.com/shibing624/pycorrector/releases/download/0.4.5/convseq2seq_correction.tar.gz

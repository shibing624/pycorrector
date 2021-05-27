# Seq2seq Attention Model


## Features

* 基于Attention机制的Sequence to Sequence模型
* Luong Attention
* Conv Seq2Seq model, GPU并行计算，训练加速
* 训练加速tricks：dataset bucketing, prefetching, token-based batching, gradients accumulation
* beam search
* chinese samples: sighan2015 sample data, CGED sample data

## Usage

### Requirements
* pip安装依赖包
```
torch>=1.4.0
transformers
tensorboardX
```

## Demo

- bertseq2seq demo

示例[seq2seq_demo.py](../../examples/seq2seq_demo.py)
```
cd ../../examples
python seq2seq_demo.py --do_train --do_predict
```

## Detail

### Preprocess


- toy train data
```
cd seq2seq
python preprocess.py
```

generate toy train data(`train.txt`) and valid data(`test.txt`), segment by char.

result:
```
# train.txt:
如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。\t如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```

### Train

```
python train.py
```

### Infer
```
python infer.py

```


### Result
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
![result image](../../docs/git_image/convseq2seq_ret.png)


### big train data

1. download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB], put data to `seq2seq/output` folder.
2. run `preprocess.py`.
```
python preprocess.py
```

# Transformer Model(Tensorflow 2.0)

## Features

* 全attention的结构代替了lstm用于解决sequence to sequence问题，语义特征提取效果更好，训练速度更快
* 基于`OpenNMT-tf`的标准Transformer model
* 训练加速tricks：dataset bucketing, prefetching, token-based batching, gradients accumulation
* beam search

## Usage

### Requirements
* pip安装依赖包
```bash
pip install tensorflow>=2.0.0 OpenNMT-tf==2.4.0
# tensorflow-gpu>=2.0.0
```

### Preprocess


- toy train data
```
cd transformer/tf
python preprocess.py
```

generate toy train data(`src-train.txt` and `tgt-train.txt`) and valid data(`src-test.txt` and `tgt-test.txt`), segment by char.

result:
```
# src-train.txt:
如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。

# tgt-train.txt:
如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```

- big train data

1. download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB], put data to `transformers/tf/output` folder.
2. run `preprocess.py`.
```
python preprocess.py
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
input: 少先队员应该给老人让坐 output: 少先队员应该给老人让座
input: 由我起开始做 output: 由我开始做

```

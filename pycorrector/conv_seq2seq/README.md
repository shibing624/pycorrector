# Neural Text Error Correction with CNN Sequence-to-Sequence Model


## Features

The model is equipped with following features:

- ```Attention based seq2seq framework.```
Encoder and decoder can be LSTM or GRU. The attention scores can be calculated with three different alignment methods.

- ```CONV seq2seq network.```

- ```Beam search algorithm.```
We implemented an efficient beam search algorithm that can also handle cases when batch_size>1.

- ```Unknown words replacement.```
This meta-algorithm can be used along with any attention based seq2seq model.
The OOV words UNK in summaries are manually replaced with words in source articles using attention weights.

## preprocess


- toy train data
```
cd conv_seq2seq
python preprocess.py

```

- big train data
```
download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB]
```

generate toy train data(`train.src` and `train.trg`) and valid data(`val.src` and `val.trg`), segment by char


```
# train.src:
吸 烟 对 人 的 健 康 有 害 处 ， 这 是 各 个 人 都 知 道 的 事 实 。
也 许 是 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 以 及 未 来 。
如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。

# train.trg:
吸 烟 对 人 的 健 康 有 害 处 ， 这 是 每 个 人 都 知 道 的 事 实 。
也 许 每 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 和 未 来 。
如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```


## train

```
sh train.sh
```

## infer
```
sh infer.sh
```

### result
```
input: 少先队员因该给老人让坐 output: 少先队员因该给老人让座
input: 少先队员应该给老人让坐 output: 少先队员应该给老人让座
input: 没有解决这个问题， output: 没有解决这个问题，，
input: 由我起开始做。 output: 由我起开始做
input: 由我起开始做 output: 由我开始做

```

### reference
1. [《基于深度学习的中文文本自动校对研究与实现》[杨宗霖, 2019]](https://github.com/shibing624/pycorrector/blob/master/docs/基于深度学习的中文文本自动校对研究与实现.pdf)
2. [《A Sequence to Sequence Learning for Chinese Grammatical Error Correction》[Hongkai Ren, 2018]](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_36)
2. [《Neural Abstractive Text Summarization with Sequence-to-Sequence Models》[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)

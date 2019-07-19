# Neural Text Error Correction with Sequence-to-Sequence Models


## Features

The model is equipped with following features:

- ```Attention based seq2seq framework.```
Encoder and decoder can be LSTM or GRU. The attention scores can be calculated with three different alignment methods.

- ```Pointer-generator network.```

- ```Intra-temporal attention mechanism and intra-decoder attention mechanism.```

- ```Coverage mechanism.```

- ```Weight sharing mechanism.```
Weight sharing mechanism can boost the performance with significantly less parameters.

- ```Beam search algorithm.```
We implemented an efficient beam search algorithm that can also handle cases when batch_size>1.

- ```Unknown words replacement.```
This meta-algorithm can be used along with any attention based seq2seq model.
The OOV words UNK in summaries are manually replaced with words in source articles using attention weights.

## preprocess
```
cd seq2seq
python preprocess.py
```
generate train data and test data, split by TAB('\t')

```
据科学理论，被动吸烟者的危害比吸烟者更厉害。	据科学理论，被动吸烟者受到的危害比吸烟者更厉害。
希望少吸烟。	希望烟民们少吸烟。
但其实禁烟这种事情非常难。	但禁烟这种事情其实非常难。
```

## train
```
python train.py
```

![train image](https://github.com/shibing624/pycorrector/blob/master/pycorrector/data/git_image/seq2seq_train.png)

## infer
```
python infer.py
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
1. [《Neural Abstractive Text Summarization with Sequence-to-Sequence Models》[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)

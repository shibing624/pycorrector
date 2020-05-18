# Transformer Model(pytorch)

## Features

* Levenshtein Transformer, Non-autoregressive Neural Machine Translation (NAT)
* random_delete noise
* share all embeddings
* Knowledge Distillation

## Usage

### Requirements
* pip安装依赖包
```bash
pip install fairseq>=0.9.0 torch>=1.3.1

```


### Preprocess


- toy train data
```
cd transformer
python preprocess.py
```

generate toy train data(`train.src` and `train.trg`) and valid data(`valid.src` and `valid.trg`), segment by char.

result:
```
# train.src:
也 许 是 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 以 及 未 来 。

# train.trg:
也 许 每 个 家 庭 都 有 子 女 而 担 心 子 女 的 现 在 和 未 来 。
```

- big train data

1. download from https://pan.baidu.com/s/1BkDru60nQXaDVLRSr7ktfA  密码:m6fg [130W sentence pair，215MB], put data to `conv_seq2seq/output` folder.
2. run `preprocess.py`.
```
python preprocess.py
```

generate fairseq format data to `bin` folder:
```
> tree transformer/output
transformer/output
├── bin
│   ├── dict.src.txt
│   ├── dict.trg.txt
│   ├── train.src-trg.src.bin
│   ├── train.src-trg.src.idx
│   ├── train.src-trg.trg.bin
│   ├── train.src-trg.trg.idx
│   ├── valid.src-trg.src.bin
│   ├── valid.src-trg.src.idx
│   ├── valid.src-trg.trg.bin
│   └── valid.src-trg.trg.idx
├── train.src
├── train.trg
├── valid.src
└── valid.trg
```

### Train

```
sh train.sh
```

### Infer
```
sh infer.sh

```

### Result
```
S-2     没 有 解 决 这 个 问 题 ， 不 能 人 类 实 现 更 美 好 的 将 来 。
T-2     没 有 解 决 这 个 问 题 ， 人 类 不 能 实 现 更 美 好 的 将 来 。
H-2     -0.6813122630119324     <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
P-2     0.0000 -0.7130 -0.7055 -0.7177 -0.7038 -0.7136 -0.7105 -0.7056 -0.7063 -0.7173 -0.7153 -0.7049 -0.7111 -0.7058 -0.7074 -0.6979 -0.7162 -0.7044 -0.7094 -0.7003 -0.7130 -0.7166 -0.7144 -0.6980 -0.7031 -0.7108 -0.7046 -0.7049 -0.7026 -0.7056 -0.6964 -0.6995 -0.7195 -0.7112 -0.7120 -0.7241 -0.7224 -0.7223 -0.7088 -0.7058 -0.7102 -0.7112 -0.7140 -0.7134 -0.7166 -0.7168 -0.7098 -0.7111 -0.7008 0.0000
I-2     1

```

## Reference
* [Levenshtein Transformer (Gu et al., 2019)](https://arxiv.org/abs/1905.11006).
* [Understanding Knowledge Distillation in Non-autoregressive Machine Translation (Zhou et al., 2019)](https://arxiv.org/abs/1911.02727).

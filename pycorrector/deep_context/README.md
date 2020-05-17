# Deep Context Model


## Features

* 基于上下文预测该未知词，类似CBOW的处理方式
* 多层级的双向lstm模型，文本语义表征能力更强
* 简单MLP预测未知词

## Usage

### Requirements
* pip安装依赖包
```
torch>=1.3.1
torchtext
```

### Preprocess


- toy train data
```
cd deep_context
python preprocess.py
```

generate toy train data(`train.txt`), segment by char.

result:
```
# train.txt:
随 着 生 活 水 平 的 提 高 ， 人 们 的 要 求 也 越 来 越 高 。
```


- big train data

中文维基百科文本均可，本质上是训练一个文本语言模型。

附上项目readme提供的人民日报2014版熟语料，网盘链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11。

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
而 且 我 希 望 不 再 存 在 抽 [] 的 人 。 [('。', -12.507756233215332), ('安', -12.577921867370605), ('下', -12.590812683105469), ('处', -12.591373443603516), ('话', -12.591511726379395), ('利', -12.59239673614502)]

```
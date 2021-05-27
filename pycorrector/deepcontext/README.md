# Deep Context Model


## Features

* 基于上下文预测该未知词，类似CBOW的处理方式
* 多层级的双向lstm模型，文本语义表征能力更强
* 简单MLP预测未知词

![framework](../../docs/git_image/framework_context.jpeg)
图片来源:https://github.com/SenticNet/context2vec

## Usage

### Requirements
* pip安装依赖包
```
pip install torch>=1.3.1
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
而 且 我 希 望 不 再 存 在 抽 [] 的 人 。 [('烟', 1.7333565950393677), ('港', 0.32165974378585815), ('题', 0.14401069283485413), ('术', 0.12335444986820221), ('染', -0.04147976636886597), ('府', -0.12270379066467285)]
```

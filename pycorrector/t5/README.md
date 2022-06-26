# T5 model for Text Correction


## Features

* 基于transformers的T5ForConditionalGeneration
* Chinese Samples: sighan2015 train data

## Usage

### Requirements
* pip安装依赖包
```
torch
transformers
loguru
```

## Demo

- T5 correction demo

example: [t5_corrector.py](t5_corrector.py)

```shell
python t5_corrector.py
```

output:
```shell
original sentence:少先队员因该为老人让坐 => 少先队员应该为老人让坐 err:[('因', '应', 4, 5)]
original sentence:少 先  队 员 因 该 为 老人让坐 => 少 先  队 员 因 该 为 老人让坐 err:[]
original sentence:机七学习是人工智能领遇最能体现智能的一个分知 => 机七学习是人工智能领域最能体现智能的一个分知 err:[('遇', '域', 10, 11)]
original sentence:今天心情很好 => 今天心情很好 err:[]
original sentence:老是较书。 => 老师教书。 err:[('是', '师', 1, 2), ('较', '教', 2, 3)]
```


### Train
data example:
```
# train.txt:
你说的是对，跟那些失业的人比起来你也算是辛运的。	你说的是对，跟那些失业的人比起来你也算是幸运的。
```
run train.py
```
python train.py --do_train --do_eval
```

### Infer
```
python infer.py
```



## Dataset

| 数据集 | 语料 | 下载链接 | 压缩包大小 |
| :------- | :--------- | :---------: | :---------: |
| **`SIGHAN+Wang271K中文纠错数据集`** | SIGHAN+Wang271K(27万条) | [百度网盘（密码01b9）](https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ)| 106M |

下载`SIGHAN+Wang271K中文纠错数据集`，解压后，为json格式。

训练示例：
```shell
python train.py --do_train --do_eval --model_name_or_path output/mengzi-t5-base-chinese-correction/ --train_path ./output/train.json --test_path output/test.json
```
## Release model
基于`SIGHAN+Wang271K中文纠错数据集`训练的T5模型，已经release到HuggingFace models:[shibing624/mengzi-t5-base-chinese-correction](https://huggingface.co/shibing624/mengzi-t5-base-chinese-correction)


### 评估结果
评估数据集：SIGHAN2015测试集

GPU：Tesla V100，显存 32 GB

| 模型 | Backbone | GPU | Precision | Recall | F1 | QPS |
| :-- | :-- | :---  | :----- | :--| :--- | :--- |
| T5 | byt5-small | GPU | 0.5220 | 0.3941 | 0.4491 | 111 |
| mengzi-t5-base-chinese-correction | mengzi-t5-base | GPU | 0.8321 | 0.6390 | 0.7229 | 214 |
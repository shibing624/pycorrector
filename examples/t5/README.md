# T5 model for Chinese Spelling Correction


## Features

* 基于transformers的T5ForConditionalGeneration
* Chinese Samples: sighan2015 train data

## Usage

### Requirements
* pip安装依赖包
```
torch
transformers
datasets
loguru
```
### 快速加载
#### pycorrector快速预测

example: [examples/t5/demo.py](https://github.com/shibing624/pycorrector/blob/master/examples/t5/demo.py)
```python
from pycorrector import T5Corrector
m = T5Corrector()
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

| 数据集 | 语料 | 下载链接 | 压缩包大小 |
| :------- | :--------- | :---------: | :---------: |
| **`SIGHAN+Wang271K中文纠错数据集`** | SIGHAN+Wang271K(27万条) | [百度网盘（密码01b9）](https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ)| 106M |

下载`SIGHAN+Wang271K中文纠错数据集`，解压后，为json格式。
### Train model
run train:
```shell
python train.py --do_train --do_eval --model_name_or_path output/mengzi-t5-base-chinese-correction/ --train_path ./output/train.json --test_path output/test.json
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


## Release model
基于`SIGHAN+Wang271K中文纠错数据集`训练的T5模型，已经release到HuggingFace models:[shibing624/mengzi-t5-base-chinese-correction](https://huggingface.co/shibing624/mengzi-t5-base-chinese-correction)


### 评估结果
评估数据集：SIGHAN2015测试集

GPU：Tesla V100，显存 32 GB

| 模型 | Backbone | GPU | Precision | Recall | F1 | QPS |
| :-- | :-- | :---  | :----- | :--| :--- | :--- |
| T5 | byt5-small | GPU | 0.5220 | 0.3941 | 0.4491 | 111 |
| mengzi-t5-base-chinese-correction | mengzi-t5-base | GPU | 0.8321 | 0.6390 | 0.7229 | 214 |

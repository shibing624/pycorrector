# LLaMA for Chinese Spelling Correction

## 简介

中文文本纠错任务是一项NLP基础任务，其输入是一个可能含有语法错误的中文句子，输出是一个正确的中文句子。
语法错误类型很多，有多字、少字、错别字等，目前最常见的错误类型是`错别字`。大部分研究工作围绕错别字这一类型进行研究。
本项目基于LLaMA实现了中文拼写纠错和语法纠错。


## 安装依赖项

- loguru
- transformers>=4.33.2
- datasets
- tqdm>=4.47.0
- accelerate>=0.21.0
- peft>=0.5.0

运行命令：
```
pip install transformers peft -U
```

## 模型训练

### 训练数据

该模型在SIGHAN简体版数据集以及[Automatic Corpus Generation生成的中文纠错数据集](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)上进行Finetune训练。PaddleNLP已经集成SIGHAN简体版数据集，以下将介绍如何使用Automatic Corpus Generation生成的中文纠错数据集。

#### 下载数据集

Automatic Corpus Generation生成的中文纠错数据集比较大，下载时间比较长，请耐心等候。运行以下命令完成数据集下载：

```
python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml
```

#### 预处理数据集

训练脚本要求训练集文件内容以句子对形式呈现，这里提供一个转换脚本，将Automatic Corpus Generation提供的XML文件转换成句子对形式的文件，运行以下命令：

```
python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt
```

### 单卡训练

```python
python train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/ --max_seq_length 192
```

### 多卡训练

```python
python -m paddle.distributed.launch --gpus "0,1"  train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/ --max_seq_length 192
```

## 模型预测

### 预测SIGHAN测试集

SIGHAN 13，SIGHAN 14，SIGHAN 15是目前中文错别字纠错任务常用的benchmark数据。由于SIGHAN官方提供的是繁体字数据集，PaddleNLP将提供简体版本的SIGHAN测试数据。以下运行SIGHAN预测脚本：

```shell
sh run_sighan_predict.sh
```

该脚本会下载SIGHAN数据集，加载checkpoint的模型参数运行模型，输出SIGHAN测试集的预测结果到predict_sighan文件，并输出预测效果。

**预测效果**

| Metric       | SIGHAN 13 | SIGHAN 14 | SIGHAN 15 |
| -------------| --------- | --------- |---------  |
| Detection F1 | 0.8348    | 0.6534    | 0.7464    |
| Correction F1| 0.8217    | 0.6302    | 0.7296    |

### 预测部署

#### 模型导出

使用动态图训练结束之后，预测部署需要导出静态图参数，具体做法需要运行模型导出脚本`export_model.py`。以下是脚本参数介绍以及运行方式：

**参数**
- `params_path` 是指动态图训练保存的参数路径。
- `output_path` 是指静态图参数导出路径。
- `pinyin_vocab_file_path` 指拼音表路径。
- `model_name_or_path` 目前支持的预训练模型有："ernie-1.0"。

**运行方式**

```shell
python export_model.py --params_path checkpoints/best_model.pdparams --output_path ./infer_model/static_graph_params
```

其中`checkpoints/best_model.pdparams`是训练过程中保存的参数文件，请更换为实际得到的训练保存路径。

#### 预测

导出模型之后，可以用于预测部署，predict.py文件提供了python预测部署示例。运行方式：

```python
python predict.py --model_file infer_model/static_graph_params.pdmodel --params_file infer_model/static_graph_params.pdiparams
```

输出如下：
```
Source: 遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
Target: 遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。
Source: 人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。
Target: 人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。
```

### pycorrector一键预测
可以使用PaddleNLP提供的Taskflow工具来对输入的文本进行一键纠错，具体使用方法如下:

```python
from paddlenlp import Taskflow
text_correction = Taskflow("text_correction", model="csc-ernie-1.0")
print(text_correction('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。'))
'''
[{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
    'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]
'''

print(text_correction('人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'))
'''
[{'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
    'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。',
    'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]
'''
```


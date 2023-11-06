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


### Dataset

#### toy data
中文语法纠错数据（1k条）：[examples/data/grammar/train_sharegpt.jsonl](https://github.com/shibing624/pycorrector/blob/master/examples/data/grammar/train_sharegpt.jsonl)

data format:
```
{"conversations":[{"from":"human","value":"对这个句子语法纠错\n\n这件事对我们大家当时震动很大。"},{"from":"gpt","value":"这件事当时对我们大家震动很大。"}]}
```


#### big train data

- 中文拼写纠错数据集：https://huggingface.co/datasets/shibing624/CSC
- 中文语法纠错数据集：https://github.com/shibing624/pycorrector/tree/llm/examples/data/grammar
- 通用GPT4问答数据集：https://huggingface.co/datasets/shibing624/sharegpt_gpt4
### Train model
run train:
```
cd examples/gpt
python train_chatglm_demo.py --do_train --do_predict
```

output:
```
input  : 这块名表带带相传
predict: 这块名表代代相传
```

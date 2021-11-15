# MacBertMaskedLM For Correction

## 使用说明

1. 下载用中文文本纠错数据集fine-tune后的预训练MACBERT CSC纠错模型（飞书文档链接: https://szuy1h04n8.feishu.cn/file/boxcnoKfHHtjokcZojQO2VjtQHB
   密码: QKz3），解压后放置于`~/.pycorrector/dataset/macbert_models/chinese_finetuned_correction`目录下。

```
macbert_models
└── chinese_finetuned_correction
    ├── config.json
    ├── added_tokens.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

2. 运行`macbert_corrector.py`进行纠错。

```
python3 macbert_corrector.py
```

3. 评估

- run
  `python tests/macbert_corrector_test.py`
- result
  ![result](../../docs/git_image/macbert_result.jpg)

纠错结果除部分英文大小写问题外，在sighan15上达到了SOTA水平。

### Evaluate

提供评估脚本[pycorrector/utils/eval.py](../utils/eval.py)，该脚本有两个功能：

- 构建评估样本集：评估集[pycorrector/data/eval_corpus.json](../data/eval_corpus.json),
  包括字粒度错误100条、词粒度错误100条、语法错误100条，正确句子200条。用户可以修改条数生成其他评估样本分布。
- 计算纠错准召率：采用保守计算方式，简单把纠错之后与正确句子完成匹配的视为正确，否则为错。

执行该评估脚本后，

MacBert模型纠错效果评估如下：

- 准确率：56.20%
- 召回率：42.67%

规则方法(加入自定义混淆集)的纠错效果评估如下：

- 准确率：320/500=64%
- 召回率：152/300=50.67%

MacBert模型在sighan15上纠错效果评估如下：

- 准确率：63.64%
- 召回率：63.64%

由于训练使用的数据使用了sighan15的训练集（复现paper使用sighan15），故在sighan15的测试集上表现较优。

## 快速加载

本项目基于pycorrector迁移的`pycorrector/transformers`，可支持BERT模型，可通过如下命令调用。当然，你也可使用官方的transformers库进行调用。

example: [correct_demo.py](correct_demo.py)

```python
from pycorrector.macbert.macbert_corrector import MacBertCorrector

model_dir = "~/.pycorrector/dataset/macbert_models/chinese_finetuned_correction"
nlp = MacBertCorrector(model_dir).macbert_correct

i = nlp('今天新情很好')
print(i)
```

如果你需要直接使用huggingface/transformers调用

1.先pip安装transformers库:

```shell
pip install transformers>=4.1.1
```
2.使用以下示例执行：

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_dir = "~/.pycorrector/dataset/macbert_models/chinese_finetuned_correction"
model = AutoModelForMaskedLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

texts = ["今天心情很好", "你找到你最喜欢的工作，我也很高心。"]
outputs = model(**tokenizer(texts, padding=True, return_tensors='pt'))
corrected_texts = []
for ids, text in zip(outputs.logits, texts):
    _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_texts.append(_text[:len(text)])

print(corrected_texts)
```

## 训练

### 安装依赖
```shell
pip install transformers>=4.1.1 pytorch-lightning>=1.1.2 torch>=1.7.0 
```
### 训练数据集

- 使用toy数据集，数据量：约1千条
```shell
cd macbert
python preprocess.py
```
得到toy数据集：
```shell
macbert/output
|-- dev.json
|-- test.json
`-- train.json
```

- [中文纠错数据集](https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ)(提取码：01b9)，数据量：约26万条，下载后新建output文件夹并放里面，文件位置同上。

数据格式：
```json
[
    {
        "id": "B2-4029-3",
        "original_text": "晚间会听到嗓音，白天的时候大家都不会太在意，但是在睡觉的时候这嗓音成为大家的恶梦。",
        "wrong_ids": [
            5,
            31
        ],
        "correct_text": "晚间会听到噪音，白天的时候大家都不会太在意，但是在睡觉的时候这噪音成为大家的恶梦。"
    },
]
```
数据集构成：
1. SIGHAN数据集，官方地址：[http://nlp.ee.ncu.edu.tw/resource/csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)
2. train.sgml数据，来源：[https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) 

### 训练
```shell
python train.py
```
### 预测
```shell
python infer.py
```

### 调用
以上即完成模型训练，把`output/macbert4csc`文件夹下以下模型文件复制到`~/.pycorrector/dataset/macbert_models/chinese_finetuned_correction`目录下，
就可以像上面说明使用pycorrector或者transformers调用。

```shell
output
└── macbert4csc
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

如果需要训练SoftMaskedBertModel，请参考[https://github.com/gitabtion/BertBasedCorrectionModels](https://github.com/gitabtion/BertBasedCorrectionModels)


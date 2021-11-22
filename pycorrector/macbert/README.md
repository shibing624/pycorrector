# MacBertMaskedLM For Correction

## 使用说明

### 快速加载

本项目是MacBERT的中文文本纠错模型，可支持BERT模型，可通过如下命令调用:

example: [correct_demo.py](correct_demo.py)

```python
from pycorrector.macbert.macbert_corrector import MacBertCorrector

nlp = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct

i = nlp('今天新情很好')
print(i)
```

当然，你也可使用官方的transformers库进行调用。

1.先pip安装transformers库:

```shell
pip install transformers>=4.1.1
```
2.使用以下示例执行：

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")

texts = ["今天心情很好", "你找到你最喜欢的工作，我也很高心。"]
outputs = model(**tokenizer(texts, padding=True, return_tensors='pt'))
corrected_texts = []
for ids, text in zip(outputs.logits, texts):
    _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_texts.append(_text[:len(text)])
print(corrected_texts)
```

模型文件组成：
```
macbert4csc-base-chinese
    ├── config.json
    ├── added_tokens.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

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

- Sentence Level: 准确率：80.98% 召回率：72.92% F1:76.74%
- Char Level: 准确率：92.47% 召回率：86.25% F1:89.25%

由于训练使用的数据使用了sighan15的训练集（复现paper使用sighan15），故在sighan15的测试集上达到SOTA水平。


## 训练

3. 评估

- run
  `python tests/macbert_corrector_test.py`
- result
  ![result](../../docs/git_image/macbert_result.jpg)

纠错结果除部分英文大小写问题外，在sighan15上达到了SOTA水平。


### 安装依赖
```shell
pip install transformers>=4.1.1 pytorch-lightning>=1.1.2 torch>=1.7.0 
```
### 训练数据集

#### toy数据集（约1千条）
```shell
cd macbert
python preprocess.py
```
得到toy数据集文件：
```shell
macbert/output
|-- dev.json
|-- test.json
`-- train.json
```
#### SIGHAN+Wang271K中文纠错数据集


| 数据集 | 语料 | 下载链接 | 压缩包大小 |
| :------- | :--------- | :---------: | :---------: |
| **`SIGHAN+Wang271K中文纠错数据集`** | SIGHAN+Wang271K(27万条) | [百度网盘（密码01b9）](https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ)| 106M |
| **`原始SIGHAN数据集`** | SIGHAN13 14 15 | [官方csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)| 339K |
| **`原始Wang271K数据集`** | Wang271K | [Automatic-Corpus-Generation dimmywang提供](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)| 93M |


SIGHAN+Wang271K中文纠错数据集，数据格式：
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

下载`SIGHAN+Wang271K中文纠错数据集`，下载后新建output文件夹并放里面，文件位置同上。

### 训练
```shell
python train.py
```
### 预测
- 方法一：直接加载保存的ckpt文件：
```shell
python infer.py
```

- 方法二：加载`pytorch_model.bin`文件：
把`output/macbert4csc`文件夹下以下模型文件复制到`~/.pycorrector/dataset/macbert_models/chinese_finetuned_correction`目录下，
就可以像上面`快速加载`使用pycorrector或者transformers调用。

```shell
output
└── macbert4csc
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

demo示例[macbert_corrector.py](macbert_corrector.py):
```
python3 macbert_corrector.py
```


如果需要训练SoftMaskedBertModel，请参考[https://github.com/gitabtion/BertBasedCorrectionModels](https://github.com/gitabtion/BertBasedCorrectionModels)


# BART文本纠错-中文-通用领域-large


#### 中文文本纠错模型介绍
输入一句中文文本，文本纠错技术对句子中存在拼写、语法、语义等错误进行自动纠正，输出纠正后的文本。主流的方法为seq2seq和seq2edits，常用的中文纠错数据集包括NLPCC18和CGED等，我们最新的工作提供了高质量、多答案的测试集MuCGEC。

我们采用基于transformer的seq2seq方法建模文本纠错任务。模型训练上，我们使用中文BART作为预训练模型，然后在Lang8和HSK训练数据上进行finetune。不引入额外资源的情况下，本模型在NLPCC18测试集上达到了SOTA。

模型效果如下：
输入：这洋的话，下一年的福气来到自己身上。
输出：这样的话，下一年的福气就会来到自己身上。

#### 期望模型使用方式以及适用范围
本模型主要用于对中文文本进行错误诊断，输出符合拼写、语法要求的文本。该纠错模型是一个句子级别的模型，模型效果会受到文本长度、分句粒度的影响，建议是每次输入一句话。具体调用方式请参考代码示例。





## Usage
#### 安装依赖
```shell
pip install pycorrector difflib modelscope==1.16.0 fairseq==0.12.2
```
#### pycorrector快速预测

example: [examples/mucgec_bart/demo.py](https://github.com/shibing624/pycorrector/blob/master/examples/mucgec_bart/demo.py)
```python
from pycorrector.mucgec_bart.mucgec_bart_corrector import MuCGECBartCorrector


if __name__ == "__main__":
    m = MuCGECBartCorrector()
    result = m.correct_batch(['这洋的话，下一年的福气来到自己身上。', 
                               '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。', 
                               '随着中国经济突飞猛近，建造工业与日俱增', 
                               "北京是中国的都。", 
                               "他说：”我最爱的运动是打蓝球“", 
                               "我每天大约喝5次水左右。", 
                               "今天，我非常开开心。"])
    print(result)
```

output:
```shell
[{'source': '今天新情很好', 'target': '今天心情很好', 'errors': [('新', '心', 2)]},
{'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}]
```

## Reference
- https://modelscope.cn/models/iic/nlp_bart_text-error-correction_chinese/summary
- 苏大：Tang et al. Chinese grammatical error correction enhanced by data augmentation from word and character levels. 2021.
- 北大 & MSRA & CUHK：Sun et al. A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-Lingual Language Model. 2021.
- Ours：Zhang et al. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction. 2022.
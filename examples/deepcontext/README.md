# Deep Context Language Model for Chinese Spelling Correction


## Features

* 基于上下文预测该未知词，类似CBOW的处理方式
* 多层级的双向lstm模型，文本语义表征能力更强
* 简单MLP预测未知词，基于`mask token predict`做token级别纠错

![framework](https://github.com/shibing624/pycorrector/blob/master/docs/git_image/framework_context.jpeg)

## Usage
### 快速加载
#### pycorrector快速预测

example: [examples/deepcontext/demo.py](https://github.com/shibing624/pycorrector/blob/master/examples/deepcontext/demo.py)
```python
from pycorrector import DeepContextCorrector
m = DeepContextCorrector()
print(m.correct_batch(['今天新情很好', '你找到你最喜欢的工作，我也很高心。']))
```

output:
```shell
[{'source': '今天新情很好', 'target': '今天心情很好', 'errors': [('新', '心', 2)]},
{'source': '你找到你最喜欢的工作，我也很高心。', 'target': '你找到你最喜欢的工作，我也很高兴。', 'errors': [('心', '兴', 15)]}]
```

### Dataset


#### toy train data
中文维基百科200条数据，见
[examples/data/wiki_zh_200.txt](https://github.com/shibing624/pycorrector/blob/master/examples/data/wiki_zh_200.txt)

#### big train data

中文维基百科文本均可，本质上是训练一个文本语言模型。


- 16GB中英文无监督、平行语料[Linly-AI/Chinese-pretraining-dataset](https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset)
- 524MB中文维基百科语料[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- 人民日报2014版熟语料，网盘链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11
### Train model

```
python train.py --do_train --do_predict
```

### Predict model
```
python predict.py
```

output:
```
input  : 老是较书。
predict: 老师教书。 [('是', '师', 1, 2), ('较', '教', 2, 3)]
```


## Reference
- https://github.com/SenticNet/context2vec
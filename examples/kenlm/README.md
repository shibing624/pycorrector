# Statistical Language Model for Chinese Spelling Correction


## Features

* ngram统计语言模型：kenlm


## Usage
### 快速加载
#### pycorrector快速预测

example: [examples/kenlm/demo.py](https://github.com/shibing624/pycorrector/blob/master/examples/kenlm/demo.py)
```python
from pycorrector import Corrector
m = Corrector()
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
参考：
https://blog.csdn.net/mingzai624/article/details/79560063?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169925331716800222836904%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=169925331716800222836904&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-79560063-null-null.nonecase&utm_term=kenlm&spm=1018.2226.3001.4450

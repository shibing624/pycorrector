# RNN Language Model

## Train BI-LSTM Language Model with Chinese Corpus

### Chinese Corpus

- 人名日报2014版数据（网盘链接:https://pan.baidu.com/s/1971a5XLQsIpL0zL0zxuK2A  密码:uc11）101MB
- CGED三年比赛数据（本项目已经提供,运行`preprocess.py`处理得到）2.8MB
- 部分中文维基百科数据（wiki上自行下载）50MB


### train

- CGED比赛数据集（小数据集）
```
python preprocess.py
python train.py
python infer.py

```
- Chinese Corpus(自定义数据集)
```bash
vim config.py train_word_path='/your_file_path.txt'
python train.py
python infer.py
```

- Result
```
output/model
├── checkpoint
├── lm-20.data-00000-of-00001
├── lm-20.index
├── lm-20.meta
```

PS:提供使用以上方法训练20轮后的中文bi-lstm语言模型，位于`output/model`目录下。

## Predict Result
- run
 `python tests/detector_test.py`
- result
待补充TODO

### 结论
与统计语言模型kenlm相比，深度rnn语言模型效果稍好，错误检测部分的误召回有降低，纠错部分依然有误纠的情况。

## 附录
- 训练时长：4块p40GPU训练20轮，超过8小时，多GPU训练部分待优化（可以使用tf.estimator.Estimator优化）。
- GPU机器配置：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           On   | 00000000:00:0A.0 Off |                    0 |
| N/A   82C    P0   172W / 250W |  22747MiB / 22919MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P40           On   | 00000000:00:0B.0 Off |                    0 |
| N/A   25C    P8     9W / 250W |     10MiB / 22919MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla P40           On   | 00000000:00:0C.0 Off |                    0 |
| N/A   26C    P8    10W / 250W |     10MiB / 22919MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

```
- chinese corpus数据示例


数据截图：
![corpus](https://github.com/shibing624/pycorrector/blob/master/pycorrector/data/git_image/peoplecorpus.png)
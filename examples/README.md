## examples

* [kenlm](kenlm)：Kenlm模型，本项目基于Kenlm统计语言模型工具训练了中文NGram语言模型，结合规则方法、混淆集可以纠正中文拼写错误，方法速度快，扩展性强，效果一般
* [deepcontext](deepcontext)：DeepContext模型，本项目基于PyTorch实现了用于文本纠错的DeepContext模型，该模型结构参考Stanford University的NLC模型，2014英文纠错比赛得第一名，效果一般
* [seq2seq](seq2seq)：Seq2Seq模型，本项目基于PyTorch实现了用于中文文本纠错的ConvSeq2Seq模型，该模型在NLPCC-2018的中文语法纠错比赛中，使用单模型并取得第三名，可以并行训练，模型收敛快，效果一般
* [t5](t5)：T5模型，本项目基于PyTorch实现了用于中文文本纠错的T5模型，使用Langboat/mengzi-t5-base的预训练模型finetune中文纠错数据集，模型改造的潜力较大，效果好
* [ernie_csc](ernie_csc)：ERNIE_CSC模型，本项目基于PaddlePaddle实现了用于中文文本纠错的ERNIE_CSC模型，模型在ERNIE-1.0上finetune，模型结构适配了中文拼写纠错任务，效果好
* [macbert](macbert)：MacBERT模型，本项目基于PyTorch实现了用于中文文本纠错的MacBERT4CSC模型，模型加入了错误检测和纠正网络，适配中文拼写纠错任务，效果好
* [gpt](gpt)：GPT模型，本项目基于PyTorch实现了用于中文文本纠错的ChatGLM/LLaMA模型，模型在中文CSC和语法纠错数据集上finetune，适配中文文本纠错任务，效果好
* [evaluate_models](evaluate_models)：模型评估，本项目基于SIGHAN2015_test数据集，对各个纠错模型进行评估
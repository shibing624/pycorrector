# Use a base image with Python3.8 version
FROM docker pull nikolaik/python-nodejs:python3.8-nodejs21-slim
MAINTAINER XuMing "xuming624@qq.com"

# install kenlm
RUN pip3 install kenlm==0.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# clone repo
#RUN git clone --depth=1 https://github.com/shibing624/pycorrector.git
#WORKDIR /home/work/pycorrector
# install requirements.txt
RUN pip3 install jieba pypinyin numpy six -i https://pypi.tuna.tsinghua.edu.cn/simple
# install pycorrector by pip3
RUN pip3 install pycorrector -i https://pypi.tuna.tsinghua.edu.cn/simple
# support chinese with utf-8
RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
ENV LC_ALL zh_CN.UTF-8

CMD /bin/bash
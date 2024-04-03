# Use a base image with Python3.8 version
FROM nikolaik/python-nodejs:python3.8-nodejs20
MAINTAINER XuMing "xuming624@qq.com"
WORKDIR /app
# install kenlm
# jieba
# pypinyin
# transformers>=4.28.1
# numpy
# pandas
# six
# loguru
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip3 install kenlm jieba pypinyin numpy six pandas loguru transformers==4.28.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# install pycorrector by pip3
RUN pip3 install pycorrector -i https://pypi.tuna.tsinghua.edu.cn/simple
# # support chinese with utf-8
# RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
# ENV LC_ALL zh_CN.UTF-8

CMD /bin/bash
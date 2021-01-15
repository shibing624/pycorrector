FROM centos:7
RUN  yum -y install python36
RUN  yum -y install git boost-devel boost-test boost zlib bzip2 xz cmake make
RUN  yum -y install gcc-c++
RUN  yum -y install python36-devel
RUN  mkdir /home/work
RUN  pip3 install https://github.com/kpu/kenlm/archive/master.zip
WORKDIR /home/work
RUN git clone https://github.com/shibing624/pycorrector.git
WORKDIR /home/work/pycorrector
RUN pip3 install -r requirements.txt
#RUN pip3 install -r requirements-dev.txt
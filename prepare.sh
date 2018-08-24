#!/usr/bin/env bash
#
set -exu

# # checkout update
git_url="https://github.com/qbetterk/pycorrector.git"
git pull $git_url


# # preparing 
echo "download and install related packages..."
pip install -r requirements.txt

# # to train the language model with kenlm toolkit
kenlm_data_path="pycorrector/data/kenlm"

# # download and process nlpcc training data 
echo "downloading lm training data( nlpcc 2018 GEC training data)..."
nlpcc_data_url="http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz"
wget -O ${kenlm_data_path}/nlpcc_data.tar.gz $nlpcc_data_url
tar -xzvf $nlpcc_data_path

echo "processing lm training data( nlpcc 2018 GEC training data)..."
awk '{print $NF}' ${kenlm_data_path}/NLPCC2018_GEC-master/Data/training/train.txt \
                > ${kenlm_data_path}/nlpcc.txt

python pycorrector/tra2sim.py -i ${kenlm_data_path}/nlpcc.txt \
                              -o ${kenlm_data_path}/nlpcc.txt \
                              -e True

python pycorrector/lm_train.py -i ${kenlm_data_path}/nlpcc.txt \
                               -o ${kenlm_data_path}/nlpcc_char.txt \
                               -c True      # set False when word segmentation is required

# # download kenlm toolkit
echo "downloading kenlm toolkit"
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
cd ../..

# # training lm
echo "training language model"
file_name="nlpcc_char"
ngram=5
# change para '-o' to set n-gram model(e.g. -o 3 as trigram)
kenlm/build/bin/lmplz -o $ngram \
                      --verbose_header \
                      --text ${kenlm_data_path}/$file_name.txt \
                      --arpa ${kenlm_data_path}/$file_name.arpa

# to compress model into a binary one for easy loading
kenlm/build/bin/build_binary ${kenlm_data_path}/$file_name.arpa \
                             ${kenlm_data_path}/$file_name.klm

echo "language_model_path = 'data/kenlm/$file_name_${ngram}gram.klm'" >> pycorrector/config.py





















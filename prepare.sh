#!/usr/bin/env bash
#
set -eu

stage=0
# whether train a new language model
train_new_lm=True
# whether use nlpcc2018gec data to train a new language model
nlpcc_lm=True
# name of file storing tokenized sentences
# the completed file name should be ${file_name}.txt
file_name="nlpcc_char"
# train a charactor level lm or a word level lm
char_level_lm=True
# n-gram model
ngram=5

# if [ $stage -le 0 ]; then
#     # # checkout update
#     git_url="https://github.com/qbetterk/pycorrector.git"
#     git pull $git_url
# fi

if [ $stage -le 1 ]; then
    # # dependency 
    echo "download and install related packages..."
    pip install -r requirements.txt
fi

if [ $stage -le 2 ] && [ $train_new_lm ]; then
    # # to train the language model with kenlm toolkit
    kenlm_data_path="pycorrector/data/kenlm"

    if [ $nlpcc_lm ]; then
        if [ ! -e ${kenlm_data_path}/nlpcc_data.tar.gz ] && \
           [ ! -d ${kenlm_data_path}/NLPCC2018_GEC_TrainingData/ ]; then
            # # download and process nlpcc training data 
            echo "downloading lm training data( nlpcc 2018 GEC training data)..."
            nlpcc_data_url="http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz"
            wget -O ${kenlm_data_path}/nlpcc_data.tar.gz $nlpcc_data_url
            tar -xzvf ${kenlm_data_path}/nlpcc_data.tar.gz -C ${kenlm_data_path}/

        elif [ -e ${kenlm_data_path}/nlpcc_data.tar.gz ]; then
            tar -xzvf ${kenlm_data_path}/nlpcc_data.tar.gz -C ${kenlm_data_path}/
        fi

        echo "processing lm training data( nlpcc 2018 GEC training data)..."
        awk '{print $NF}' ${kenlm_data_path}/NLPCC2018_GEC_TrainingData/data.train \
                    > ${kenlm_data_path}/nlpcc.txt                


        python pycorrector/tra2sim.py -i ${kenlm_data_path}/nlpcc.txt \
                                      -o ${kenlm_data_path}/nlpcc.txt \
                                      -e True

        python pycorrector/lm_train.py -i ${kenlm_data_path}/nlpcc.txt \
                                       -o ${kenlm_data_path}/${file_name}.txt \
                                       -c $char_level_lm      
    fi

    if [ ! -d kenlm ]; then
        # # download kenlm toolkit
        echo "downloading kenlm toolkit"
        git clone https://github.com/kpu/kenlm.git
        cd kenlm
        mkdir -p build
        cd build
        cmake ..
        make -j 4
        cd ../..
    fi

    # # training lm
    echo "training language model"
    kenlm/build/bin/lmplz -o $ngram \
                          --verbose_header \
                          --text ${kenlm_data_path}/${file_name}.txt \
                          --arpa ${kenlm_data_path}/${file_name}_${ngram}gram.arpa

    # to compress model into a binary one for easy loading
    kenlm/build/bin/build_binary ${kenlm_data_path}/${file_name}_${ngram}gram.arpa \
                                 ${kenlm_data_path}/${file_name}_${ngram}gram.klm

    if [ $uname = Darwin ]; then
        sed -i '' 's/^language_model_path/# language_model_path/' pycorrector/config.py
    else
        sed -i 's/^language_model_path/# language_model_path/' pycorrector/config.py
    fi

    echo "language_model_path = 'data/kenlm/${file_name}_${ngram}gram.klm'" >> pycorrector/config.py
fi 

echo "successful preparing"



















# pycorrector

## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Building pycorrector](#building-pycorrector)
* [Running pycorrector](#running-pycorrector)
* [Example](#examples)
   * [Dataset](#dataset)
   * [Running script](#running-scripts)
* [Full documentation](#full-documentation)
* [License](#license)

## Introduction

[pycorrector](https://fasttext.cc/) is a toolkit for Chinese spelling error detection and correction with statistical methods.


## Requirements

Generally, **pycorrector** builds on modern Mac OS (#and Linux) distributions.

For the dependencies you will need:

* Python 3.6 or newer
* NumPy
* kenlm
* jieba
* pypinyin
* cmake
* tqdm

These dependencies are all concluded in requirements.txt and you could easily install them with python install command 'pip' like:

```
$ pip install -r requirements.txt
```

A conda environment is suggested to avoid python version conflict.


## Building pycorrector

```
$ git clone https://github.com/qbetterk/pycorrector.git
$ cd pycorrector
$ pip install -r requirements.txt
$ bash prepare.sh
```

The default setting in `prepare.sh` in to mainly download kenlm toolkit and dataset from NLPCC2018 GEC shared task to train a charactor based language model.
If you want to have your own language model based on your dataset, you could add `-train_new_lm=False` to command `base prepare.sh`. 
Train a statistical language model with kenlm(https://github.com/kpu/kenlm) is required:
```
# get kenlm source code and compile it 
$ git clone https://github.com/kpu/kenlm.git
$ cd kenlm
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j 4

# training language model based on your own training data
$ ./build/bin/lmplz -o 3 --verbose_header --text ./train_data.txt --arpa ./train_data.arps
$ ./build/bin/build_binary ./train_data.arps ./train_data.klm
```

Note that the train_data.txt is required to be tokenized(word level or charactor level) and remember to add relative path of `train_data.klm` from `pycorrector/` to `pycorrecor/config.py`.



## Running pycorrector

For error detection:
```
$ python pycorrector/detector.py
```

For error correction:
```
$ python pycorrector/corrector.py
```

See more available arguments in [Full documentation](#full-documentation) part.


## Example

Here we take SIGHAN 2015 Bake-off: Chinese Spelling Check Task as an example.


### Dataset

Considering the copyright, you have to download the dataset manually on SIGHAN 2015 website(http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)

Please uncompress the dataset under the folder `sighan8_test/`


### Running script

Runnning the example by command:

```
$ cd sighan8_test/
$ bash sighan8_test.sh
```

The score should be stored in `sighan8_result/sighan15_evaluation_test.txt`, the readable result, containing comparasion between original sentences and predicted sentences, is stored in `sighan8_result/result_compare.tmp`. There are also three more result file containing only false readable result, true positive results and true negative results.


## Full documentation

Invoke a command without arguments to list available arguments and their default values:

Error detection:
```
$ python ./pycorrector/corrector.py

The following arguments are mandatory( not if you would like to correct sentence one by one):
  -i                  file path to error sentences
  -o                  file path to detected errors(output)

The following arguments are optional:
  -v                  show detecting detail of each sentence[False]
```

Error correction
```
$ python ./pycorrector/corrector.py

The following arguments are mandatory( not if you would like to correct sentence one by one):
  -i                  file path to error sentences
  -o                  file path to corrected sentences(output)

The following arguments are optional:
  -v                  show correcting detail [False]

The following arguments for the tunning are optional(you might want to tune these depending on your training data):
  --param_ec          parameter for adjust the weight of edition cost [1.4]
  --param_gd          parameter for adjust the weight of global decision [2]

```

## License

fastText is BSD-licensed. We also provide an additional patent grant.
# Copyright 2016 Stanford University
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

# Special vocabulary symbols - we always put them at the start.
_PAD = "<pad>"
_SOS = "<sos>"
_EOS = "<eos>"
_UNK = "<unk>"
START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    """Very basic bert_tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def char_tokenizer(sentence):
    return list(sentence.strip())


def bpe_tokenizer(sentence):
    tokens = sentence.strip().split()
    tokens = [w + "</w>" if not w.endswith("@@") else w for w in tokens]
    tokens = [w.replace("@@", "") for w in tokens]
    return tokens


def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        # for path in data_paths:
        with open(data_paths, mode="r", encoding='utf-8') as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 1000 == 0:
                    print("  processing line %d" % counter)
                # Remove non-ASCII characters
                # line = remove_nonascii(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    # word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    word = w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w", encoding='utf-8') as vocab_file:
            for w in vocab_list:
                vocab_file.write(str(w) + "\n")


def initialize_vocabulary(vocabulary_path, bpe=False):
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="r", encoding='utf-8') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.rstrip('\n') for line in rev_vocab]
        # Call ''.join below since BPE outputs split pairs with spaces
        if bpe:
            vocab = dict([(''.join(x.split(' ')), y) for (y, x) in enumerate(rev_vocab)])
        else:
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, r"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
    if not os.path.exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path, bpe=(tokenizer == bpe_tokenizer))
        with open(data_path, mode="r", encoding='utf-8') as data_file:
            with open(target_path, mode="w", encoding='utf-8') as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 1000 == 0:
                        print("  tokenizing line %d" % counter)
                    # line = remove_nonascii(line)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
            print('data {0} write to {1}'.format(data_path, target_path))


def prepare_nlc_data(data_dir, vocabulary_size, tokenizer=char_tokenizer):
    """Get NLC data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      vocabulary_size: size of the English vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for English training data-set,
        (2) path to the token-ids for French training data-set,
        (3) path to the token-ids for English development data-set,
        (4) path to the token-ids for French development data-set,
        (5) path to the vocabulary file,
    """
    # Get nlc data to the specified directory.
    train_path = os.path.join(data_dir, 'train')
    dev_path = os.path.join(data_dir, 'test')

    # Create vocabularies of the appropriate sizes.
    # y_vocab_path = os.path.join(data_dir, "vocab%d.y" % y_vocabulary_size)
    vocab_path = os.path.join(data_dir, "vocab.txt")
    create_vocabulary(vocab_path, train_path + ".y.txt", vocabulary_size, tokenizer)
    create_vocabulary(vocab_path, train_path + ".x.txt", vocabulary_size, tokenizer)

    # Create token ids for the training data.
    y_train_ids_path = train_path + (".ids%d.y" % vocabulary_size)
    x_train_ids_path = train_path + (".ids%d.x" % vocabulary_size)
    data_to_token_ids(train_path + ".y.txt", y_train_ids_path, vocab_path, tokenizer)
    data_to_token_ids(train_path + ".x.txt", x_train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    y_dev_ids_path = dev_path + (".ids%d.y" % vocabulary_size)
    x_dev_ids_path = dev_path + (".ids%d.x" % vocabulary_size)
    data_to_token_ids(dev_path + ".y.txt", y_dev_ids_path, vocab_path, tokenizer)
    data_to_token_ids(dev_path + ".x.txt", x_dev_ids_path, vocab_path, tokenizer)

    return (x_train_ids_path,
            y_train_ids_path,
            x_dev_ids_path,
            y_dev_ids_path,
            vocab_path)

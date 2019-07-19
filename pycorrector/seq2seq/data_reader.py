# -*- coding: utf-8 -*-
# Author: Tian Shi <tshi@vt.edu>, XuMing <xuming624@qq.com>
# Brief: Corpus for model
import os
import random
import shutil
import sys
from codecs import open
from collections import Counter

# Define constants associated with the usual special tokens.
PAD_ID = 0
GO_ID = 1
END_ID = 2
UNK_ID = 3

PAD_TOKEN = 'PAD'
GO_TOKEN = 'GO'
END_TOKEN = 'END'
UNK_TOKEN = 'UNK'


def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                print('error', line)
    return dict_data


def create_batch_file(output_dir, file_type, file_path, batch_size=8):
    """
    Split the corpus into batches.
    :param output_dir:
    :param file_type:
    :param file_path:
    :param batch_size:
    :return:
    """
    folder = os.path.join(output_dir, 'batch_' + file_type + '_' + str(batch_size))

    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    except:
        os.mkdir(folder)

    corpus_arr = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_arr.append(line.lower())
    if file_type == 'train' or file_type == 'validate':
        random.shuffle(corpus_arr)

    cnt = 0
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            with open(os.path.join(folder, str(cnt)), 'w', encoding='utf-8') as f:
                for sen in arr:
                    f.write(sen)
            arr = []
            cnt += 1

    if len(arr) > 0:
        with open(os.path.join(folder, str(cnt)), 'w', encoding='utf-8') as f:
            for sen in arr:
                f.write(sen)
        arr = []
        cnt += 1

    return cnt


def process_minibatch_explicit(batch_id,
                               output_dir,
                               file_type,
                               batch_size,
                               vocab2id,
                               max_lens=[128, 128]):
    """
    Process the mini batch. OOV explicit.
    :param batch_id:
    :param output_dir:
    :param file_type:
    :param batch_size:
    :param vocab2id:
    :param max_lens:
    :return:
    """
    file_path = os.path.join(output_dir, 'batch_' + file_type + '_' + str(batch_size), str(batch_id))
    # build extended vocabulary
    ext_vocab = {}
    ext_id2oov = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            wrongs = list(parts[0])
            wrongs = list(filter(None, wrongs))
            for wd in wrongs:
                if wd not in vocab2id:
                    ext_vocab[wd] = {}
            rights = list(parts[1])
            rights = list(filter(None, rights))
            for wd in rights:
                if wd not in vocab2id:
                    ext_vocab[wd] = {}
        cnt = len(vocab2id)
        for wd in ext_vocab:
            ext_vocab[wd] = cnt
            ext_id2oov[cnt] = wd
            cnt += 1

    src_arr = []
    src_arr_ex = []
    trg_arr = []
    trg_arr_ex = []
    src_lens = []
    trg_lens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 2:
                print("error line, part len not 2. ", line)
                continue
            # trg
            rights = list(parts[1])
            rights = list(filter(None, rights)) + [END_TOKEN]
            trg_lens.append(len(rights))
            # UNK
            right2id = [
                vocab2id[wd] if wd in vocab2id
                else vocab2id[UNK_TOKEN]
                for wd in rights
            ]
            trg_arr.append(right2id)
            # extend vocab
            right2id = [
                vocab2id[wd] if wd in vocab2id
                else ext_vocab[wd]
                for wd in rights
            ]
            trg_arr_ex.append(right2id)

            # src
            wrongs = list(parts[0])
            wrongs = list(filter(None, wrongs))
            src_lens.append(len(wrongs))
            # UNK
            wrong2id = [
                vocab2id[wd] if wd in vocab2id
                else vocab2id[UNK_TOKEN]
                for wd in wrongs
            ]
            src_arr.append(wrong2id)
            # extend vocab
            wrong2id = [
                vocab2id[wd] if wd in vocab2id
                else ext_vocab[wd]
                for wd in wrongs
            ]
            src_arr_ex.append(wrong2id)

    src_max_lens = max_lens[0]
    trg_max_lens = max_lens[1]

    src_arr = [itm[:src_max_lens] for itm in src_arr]
    trg_arr = [itm[:trg_max_lens] for itm in trg_arr]
    src_arr_ex = [itm[:src_max_lens] for itm in src_arr_ex]
    trg_arr_ex = [itm[:trg_max_lens] for itm in trg_arr_ex]

    src_arr = [
        itm + [vocab2id[PAD_TOKEN]] * (src_max_lens - len(itm))
        for itm in src_arr
    ]
    trg_input_arr = [
        itm[:-1] + [vocab2id[PAD_TOKEN]] * (1 + trg_max_lens - len(itm))
        for itm in trg_arr
    ]
    # extend oov
    src_arr_ex = [
        itm + [vocab2id[PAD_TOKEN]] * (src_max_lens - len(itm))
        for itm in src_arr_ex
    ]
    trg_output_arr_ex = [
        itm[1:] + [vocab2id[PAD_TOKEN]] * (1 + trg_max_lens - len(itm))
        for itm in trg_arr_ex
    ]

    return ext_id2oov, src_arr, trg_input_arr, src_arr_ex, trg_output_arr_ex


def process_minibatch_explicit_test(batch_id,
                                    output_dir,
                                    batch_size,
                                    vocab2id,
                                    src_lens):
    """
    Process the minibatch test. OOV explicit.
    :param batch_id: 
    :param output_dir: 
    :param batch_size: 
    :param vocab2id: 
    :param src_lens: 
    :return: 
    """
    file_path = os.path.join(output_dir, 'batch_test_' + str(batch_size), str(batch_id))
    # build extended vocabulary
    ext_vocab = {}
    ext_id2oov = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            wrongs = list(parts[0])
            wrongs = list(filter(None, wrongs))

            for wd in wrongs:
                if wd not in vocab2id:
                    ext_vocab[wd] = {}
        cnt = len(vocab2id)
        for wd in ext_vocab:
            ext_vocab[wd] = cnt
            ext_id2oov[cnt] = wd
            cnt += 1

    src_arr = []
    src_idx = []
    src_idx_ex = []
    src_wt = []
    trg_arr = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 2:
                print("error line, part len not 2. ", line)
                continue
            wrongs = list(parts[0])
            wrongs = list(filter(None, wrongs))
            src_arr.append(wrongs)

            rights = list(parts[1])
            rights = list(filter(None, rights))
            trg_arr.append(' '.join(rights))

            wrong2id = [vocab2id[wd] if wd in vocab2id else vocab2id[UNK_TOKEN] for wd in wrongs]
            src_idx.append(wrong2id)
            wrong2id = [vocab2id[wd] if wd in vocab2id else ext_vocab[wd] for wd in wrongs]
            src_idx_ex.append(wrong2id)
            wrong2wt = [0.0 if wd in vocab2id else 1.0 for wd in wrongs]
            src_wt.append(wrong2wt)

    src_idx = [itm[:src_lens] for itm in src_idx]
    src_var = [itm + [vocab2id[PAD_TOKEN]] * (src_lens - len(itm)) for itm in src_idx]

    src_idx_ex = [itm[:src_lens] for itm in src_idx_ex]
    src_var_ex = [itm + [vocab2id[PAD_TOKEN]] * (src_lens - len(itm)) for itm in src_idx_ex]

    src_wt = [itm[:src_lens] for itm in src_wt]
    src_msk = [itm + [0.0] * (src_lens - len(itm)) for itm in src_wt]
    # src_msk = Variable(torch.FloatTensor(src_wt))

    src_arr = [itm[:src_lens] for itm in src_arr]
    src_arr = [itm + [PAD_TOKEN] * (src_lens - len(itm)) for itm in src_arr]

    return ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr


def read_vocab(input_texts, max_size=50000, min_count=5):
    token_counts = Counter()
    special_tokens = [PAD_TOKEN, GO_TOKEN, END_TOKEN, UNK_TOKEN]
    for line in input_texts:
        for char in line.strip():
            char = char.strip()
            if not char:
                continue
            token_counts.update(char)
    # Sort word count by value
    count_pairs = token_counts.most_common()
    vocab = [k for k, v in count_pairs if v >= min_count]
    # Insert the special tokens to the beginning
    vocab[0:0] = special_tokens
    full_token_id = list(zip(vocab, range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    return vocab2id


def read_samples_by_string(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.lower().strip().split('\t')
            if len(parts) != 2:
                print('error ', line)
                continue
            source, target = parts[0], parts[1]
            yield source, target


def build_dataset(path):
    print('Read data, path:{0}'.format(path))
    sources, targets = [], []
    for source, target in read_samples_by_string(path):
        sources.append(source)
        targets.append(target)
    return sources, targets


def show_progress(curr, total, time=""):
    prog_ = int(round(100.0 * float(curr) / float(total)))
    dstr = '[' + '>' * int(round(prog_ / 4)) + ' ' * (25 - int(round(prog_ / 4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time + '\r')
    sys.stdout.flush()

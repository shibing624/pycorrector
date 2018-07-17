# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import random
import argparse
import re
from os.path import join as pjoin

'''
convert lang8 corpus into 2 text files containing parallel verb phrases
(original and corrected).

about half the examples don't have corrections; can either ignore or add those as identity mappings

currently doesn't do additional preprocessing besides that done by tajiri et al.
'''


def parse_lang8_entries(lang8_filepath):
    with open(lang8_filepath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    par_count = 0
    original = list()
    corrected = list()
    for line in lines:
        cols = line.split('\t')
        if len(cols) < 6:
            continue
        else:
            # NOTE prints examples with multiple corrections
            # if len(cols) > 6:
            # print(' ||| '.join(cols[4:]))
            # NOTE not using multiple corrections to avoid train and dev having same source sentences
            for corr_sent in cols[5:6]:
                if cols[4] == corr_sent:
                    continue
                original.append(cols[4])
                corrected.append(corr_sent)
                par_count += 1
    return original, corrected


def prep_nucle_sent(sent):
    # nucle and lang-8 tokenization differ
    sent = sent.strip()
    sent = re.sub(r'(\S)-(\S)', r'\1 - \2', sent)
    sent = re.sub(r'(\S)/(\S)', r'\1 / \2', sent)
    sent = re.sub(r'(\S),(\S)', r'\1 , \2', sent)
    sent = re.sub(r'(\S)\.(\S)', r'\1 . \2', sent)
    sent = re.sub(r'(\S)"', r'\1 "', sent)
    sent = re.sub(r'"(\S)', r'" \1', sent)
    return sent


def parse_nucle_entries(nucle_filepath):
    with open(nucle_filepath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    original = list()
    corrected = list()

    for line in lines:
        if line.startswith('S '):
            ref = line.split(' ')[1:]
            curr_sent = list(ref)
            offset = 0
        elif line.startswith('A '):
            edit = line.split(' ', 1)[1]
            inds, rest = edit.split('|||', 1)
            inds = inds.split(' ')
            inds = list(map(int, inds))
            sub = rest.split('|||')[1]
            if inds[0] == inds[1]:
                # insertion
                # assert sub

                if sub:
                    curr_sent.insert(inds[0] + offset, sub)
                    offset = offset + 1
            elif not sub:
                # deletion
                curr_sent = curr_sent[:inds[0] + offset] + curr_sent[inds[1] + offset:]
                offset = offset - (inds[1] - inds[0])
            else:
                # substitution
                curr_sent = curr_sent[:inds[0] + offset] + [sub] + curr_sent[inds[1] + offset:]
                offset = offset - (inds[1] - inds[0]) + 1
        else:
            ref = ' '.join(ref)
            curr_sent = ' '.join(curr_sent)
            ref = prep_nucle_sent(ref)
            curr_sent = prep_nucle_sent(curr_sent)
            original.append(ref)
            corrected.append(curr_sent)
    # last sentence
    original.append(' '.join(ref))
    corrected.append(' '.join(curr_sent))
    return original, corrected


def write_lines(filename, lines):
    with open(filename, 'w') as fout:
        fout.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang8_path', help='path to directory containing lang-8 entries.[split] files')
    parser.add_argument('nucle32_path', help='path to 2014 conll train m2 file')
    parser.add_argument('--dev_split_fract', default=0.01, type=float,
                        help='fraction of lang8 training data to use for dev split (e.g. 0.01)')
    args = parser.parse_args()

    # get conll training data
    nucle_m2_file = pjoin(args.nucle32_path, 'data', 'conll14st-preprocessed.m2')
    conll_train_original, conll_train_corrected = parse_nucle_entries(nucle_m2_file)

    # conll train is partitioned into conll_train and conll_test
    # lang8_train_filename = pjoin(args.lang8_path, 'entries.train')
    # lang8_train_original, lang8_train_corrected = parse_lang8_entries(lang8_train_filename)
    ntrain = int(len(conll_train_original) * (1 - args.dev_split_fract))
    dev_original, dev_corrected = conll_train_original[ntrain:], conll_train_corrected[ntrain:]
    # lang8_train_original, lang8_train_corrected = lang8_train_original[:ntrain], lang8_train_corrected[:ntrain]

    # test = lang8_test
    # lang8_test_filname = pjoin(args.lang8_path, 'entries.test')
    test_original, test_corrected = dev_original, dev_corrected

    # train = lang8_train + conll_train
    # train_original = lang8_train_original + conll_train_original
    # train_corrected = lang8_train_corrected + conll_train_corrected

    train_original = conll_train_original
    train_corrected = conll_train_corrected

    # shuffle training data
    random.seed(1234)
    combined = zip(train_original, train_corrected)
    random.shuffle(combined)
    train_original, train_corrected = zip(*combined)

    print('writing %d training examples, %d dev examples, %d test examples' % \
          (len(train_original), len(dev_original), len(test_original)))

    write_lines('data/train.x.txt', train_original)
    write_lines('data/train.y.txt', train_corrected)
    write_lines('data/dev.x.txt', dev_original)
    write_lines('data/dev.y.txt', dev_corrected)
    write_lines('data/test.x.txt', test_original)
    write_lines('data/test.y.txt', test_corrected)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: Run BERT on Masked LM.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('../..')
import argparse
import os
import random
import re

import numpy as np
import torch
from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pycorrector.utils.logger import logger

MASK_TOKEN = "[MASK]"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,
                 mask_ids=None, mask_positions=None, input_tokens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.mask_positions = mask_positions


def read_lm_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(guid=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def read_lm_sentence(sentence):
    """Read a list of `InputExample`s from an input line."""
    examples = []
    unique_id = 0
    line = sentence.strip()
    if line:
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
            InputExample(guid=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def is_subtoken(x):
    return x.startswith("##")


def create_masked_lm_prediction(input_ids, mask_position, mask_count=1, mask_id=103):
    new_input_ids = list(input_ids)
    masked_lm_labels = []
    masked_lm_positions = list(range(mask_position, mask_position + mask_count))
    for i in masked_lm_positions:
        new_input_ids[i] = mask_id
        masked_lm_labels.append(input_ids[i])
    return new_input_ids, masked_lm_positions, masked_lm_labels


def create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids, mask_id=103, tokenizer=None):
    """Mask each token/word sequentially"""
    features = []
    i = 1
    while i < len(input_tokens) - 1:
        mask_count = 1
        while is_subtoken(input_tokens[i + mask_count]):
            mask_count += 1

        input_ids_new, masked_lm_positions, masked_lm_labels = create_masked_lm_prediction(input_ids, i, mask_count,
                                                                                           mask_id)
        feature = InputFeatures(
            input_ids=input_ids_new,
            input_mask=input_mask,
            segment_ids=segment_ids,
            mask_ids=masked_lm_labels,
            mask_positions=masked_lm_positions,
            input_tokens=tokenizer.convert_ids_to_tokens(input_ids_new))
        features.append(feature)
        i += mask_count
    return features


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 mask_token='[MASK]', mask_id=103):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    all_features = []
    all_tokens = []
    for (example_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # The -3 accounts for [CLS], [SEP] and [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # "-2" is [CLS] and [SEP]
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0     0     0      0   0    1  1  1  1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0      0   0   0  0    0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens_a = [i.replace('*', mask_token) for i in tokens_a]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens_b = [i.replace('*', '[MASK]') for i in tokens_b]
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        mask_positions = [i for i, v in enumerate(input_ids) if v == mask_id]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("example_index: %s" % (example_index))
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          mask_positions=mask_positions,
                          segment_ids=segment_ids,
                          input_tokens=tokens))
        # Mask each word
        # features = create_sequential_mask(tokens, input_ids, input_mask, segment_ids, mask_id, tokenizer)
        # all_features.extend(features)
        # all_tokens.extend(tokens)
        # return all_features, all_tokens

    return features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model_dir", default='../data/bert_models/chinese_finetuned_lm/',
                        type=str,
                        help="Bert pre-trained model config dir")
    parser.add_argument("--bert_model_vocab", default='../data/bert_models/chinese_finetuned_lm/vocab.txt',
                        type=str,
                        help="Bert pre-trained model vocab path")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--predict_file", default='../data/cn/lm_test_zh.txt', type=str,
                        help="for predictions.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(args.bert_model_vocab)
    MASK_ID = tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
    print('MASK_ID,', MASK_ID)

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model_dir)

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if not os.path.exists(output_model_file):
        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model.to(device)

    # Tokenized input
    text = "吸烟的人容易得癌症"
    tokenized_text = tokenizer.tokenize(text)
    print(text, '=>', tokenized_text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Convert inputs to PyTorch tensors
    print('tokens, segments_ids:', indexed_tokens, segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    model.eval()
    # Predict all tokens
    predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    print(predicted_index)
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)
    # infer one line end

    # predict ppl and prob of each word
    text = "吸烟的人容易得癌症"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    sentence_loss = 0.0
    sentence_count = 0
    for idx, label in enumerate(text):
        print(label)
        label_id = tokenizer.convert_tokens_to_ids([label])[0]
        lm_labels = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        if idx != 0:
            lm_labels[idx] = label_id
        if idx == 1:
            lm_labels = indexed_tokens
        print(lm_labels)
        masked_lm_labels = torch.tensor([lm_labels])

        # Predict all tokens
        loss = model(tokens_tensor, segments_tensors, masked_lm_labels=masked_lm_labels)
        print('loss:', loss)
        prob = float(np.exp(-loss.item()))
        print('prob:', prob)
        sentence_loss += prob
        sentence_count += 1
    ppl = float(np.exp(sentence_loss / sentence_count))
    print('ppl:', ppl)

    # confirm we were able to predict 'henson'
    # infer each word with mask one
    text = "吸烟的人容易得癌症"
    for masked_index, label in enumerate(text):
        tokenized_text = tokenizer.tokenize(text)
        print(text, '=>', tokenized_text)
        tokenized_text[masked_index] = '[MASK]'
        print(tokenized_text)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        predictions = model(tokens_tensor, segments_tensors)
        print('expected label:', label)

        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print('predict label:', predicted_token)

        scores = predictions[0, masked_index]
        # predicted_index = torch.argmax(scores).item()
        top_scores = torch.sort(scores, 0, True)
        top_score_val = top_scores[0][:5]
        top_score_idx = top_scores[1][:5]
        for j in range(len(top_score_idx)):
            print('Mask predict is:', tokenizer.convert_ids_to_tokens([top_score_idx[j].item()])[0],
                  ' prob:', top_score_val[j].item())
        print()

    if args.predict_file:
        eval_examples = read_lm_examples(input_file=args.predict_file)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            mask_token=MASK_TOKEN,
            mask_id=MASK_ID)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("Start predict ...")
        for f in eval_features:
            input_ids = torch.tensor([f.input_ids])
            segment_ids = torch.tensor([f.segment_ids])
            predictions = model(input_ids, segment_ids)
            # confirm we were able to predict 'henson'
            mask_positions = f.mask_positions

            if mask_positions:
                for idx, i in enumerate(mask_positions):
                    if not i:
                        continue
                    scores = predictions[0, i]
                    # predicted_index = torch.argmax(scores).item()
                    top_scores = torch.sort(scores, 0, True)
                    top_score_val = top_scores[0][:5]
                    top_score_idx = top_scores[1][:5]
                    # predicted_prob = predictions[0, i][predicted_index].item()
                    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                    print('original text is:', f.input_tokens)
                    # print('Mask predict is:', predicted_token, ' prob:', predicted_prob)
                    for j in range(len(top_score_idx)):
                        print('Mask predict is:', tokenizer.convert_ids_to_tokens([top_score_idx[j].item()])[0],
                              ' prob:', top_score_val[j].item())


if __name__ == "__main__":
    main()

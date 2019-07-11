# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: use bert detect chinese char error
"""
import glob
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pycorrector.detector import ErrorType

sys.path.append('../..')
from pycorrector.bert import modeling
from pycorrector.utils.logger import logger

MASK_TOKEN = "[MASK]"
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'bert_model.ckpt'


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, mask_ids=None, mask_positions=None, input_tokens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.mask_positions = mask_positions


class InputExample(object):
    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text


class BertDetector(object):
    def __init__(self, bert_model_dir='',
                 bert_model_vocab='',
                 max_seq_length=128,
                 predict_batch_size=8,
                 max_predictions_per_seq=20,
                 threshold=0.001):
        self.name = 'bert_detector'
        self.bert_model_dir = bert_model_dir
        self.bert_model_vocab = bert_model_vocab
        self.max_seq_length = max_seq_length
        self.predict_batch_size = predict_batch_size
        self.initialized_bert_detector = False
        self.max_predictions_per_seq = max_predictions_per_seq
        self.threshold = threshold

    def check_bert_detector_initialized(self):
        if not self.initialized_bert_detector:
            self.initialize_bert_detector()

    def initialize_bert_detector(self):
        t1 = time.time()
        self.bert_tokenizer = BertTokenizer(self.bert_model_vocab)
        self.MASK_ID = self.bert_tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
        # Prepare model
        bert_config_file = os.path.join(self.bert_model_dir, BERT_CONFIG_NAME)
        if not bert_config_file:
            bert_config_file = glob.glob(self.bert_model_dir + '/*.json')[0]
        self.bert_checkpoint = os.path.join(self.bert_model_dir, TF_WEIGHTS_NAME)
        if not self.bert_checkpoint:
            self.bert_checkpoint = glob.glob(self.bert_model_dir + '/*.meta')[0][:-5]
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        model_fn = self._model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.bert_checkpoint)

        # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
        self.model = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=tf.contrib.tpu.RunConfig(),
            predict_batch_size=self.predict_batch_size)
        logger.debug("Loaded model ok, path: %s, spend: %.3f s." % (self.bert_model_dir, time.time() - t1))
        self.initialized_bert_detector = True

    @staticmethod
    def _read_examples(sentences):
        """Read a list of `InputExample`s from an input sentences."""
        examples = []
        unique_id = 0
        for sentence in sentences:
            line = sentence.strip()
            examples.append(InputExample(unique_id, line))
            unique_id += 1
        return examples

    def _model_fn_builder(self, bert_config, init_checkpoint):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=False)

            masked_lm_example_loss = self._get_masked_lm_output(
                bert_config, model.get_sequence_output(), model.get_embedding_table(),
                masked_lm_positions, masked_lm_ids)

            tvars = tf.trainable_variables()
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            output_spec = None
            if mode == tf.estimator.ModeKeys.PREDICT:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=masked_lm_example_loss, scaffold_fn=scaffold_fn)  # 输出mask_word的score
            return output_spec

        return model_fn

    def _get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                              label_ids):
        """Get loss and log probs for the masked LM."""
        input_tensor = self._gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
        return loss

    @staticmethod
    def _gather_indexes(sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    @staticmethod
    def _input_fn_builder(features, seq_length, max_predictions_per_seq):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_masked_lm_positions = []
        all_masked_lm_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_masked_lm_positions.append(feature.mask_positions)
            all_masked_lm_ids.append(feature.mask_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]
            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "segment_ids":
                    tf.constant(
                        all_segment_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "masked_lm_positions":
                    tf.constant(
                        all_masked_lm_positions,
                        shape=[num_examples, max_predictions_per_seq],
                        dtype=tf.int32),
                "masked_lm_ids":
                    tf.constant(
                        all_masked_lm_ids,
                        shape=[num_examples, max_predictions_per_seq],
                        dtype=tf.int32)
            })

            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d

        return input_fn

    # This function is not used by this file but is still used by the Colab and
    # people who depend on it.
    def _convert_examples_to_features(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        all_features = []
        all_tokens = []

        for (ex_index, example) in enumerate(examples):
            features, tokens = self._convert_single_example(example)
            all_features.extend(features)
            all_tokens.extend(tokens)

        return all_features, all_tokens

    def _convert_single_example(self, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        self.check_bert_detector_initialized()
        tokens = self.bert_tokenizer.tokenize(example.text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]

        input_tokens = []
        segment_ids = []
        input_tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens:
            input_tokens.append(token)
            segment_ids.append(0)
        input_tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(input_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length

        features = self._create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids)
        return features, input_tokens

    @staticmethod
    def is_subtoken(x):
        return x.startswith("##")

    def _create_sequential_mask(self, input_tokens, input_ids, input_mask, segment_ids):
        """Mask each token/word sequentially"""
        features = []
        i = 1
        while i < len(input_tokens) - 1:
            mask_count = 1
            while self.is_subtoken(input_tokens[i + mask_count]):
                mask_count += 1

            input_ids_new, masked_lm_positions, masked_lm_labels = self._create_masked_lm_prediction(input_ids, i,
                                                                                                     mask_count)
            while len(masked_lm_positions) < self.max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_labels.append(0)

            feature = InputFeatures(
                input_ids=input_ids_new,
                input_mask=input_mask,
                segment_ids=segment_ids,
                mask_positions=masked_lm_positions,
                mask_ids=masked_lm_labels)
            features.append(feature)
            i += mask_count
        return features

    def _create_masked_lm_prediction(self, input_ids, mask_position, mask_count=1):
        new_input_ids = list(input_ids)
        masked_lm_labels = []
        masked_lm_positions = list(range(mask_position, mask_position + mask_count))
        for i in masked_lm_positions:
            new_input_ids[i] = self.MASK_ID
            masked_lm_labels.append(input_ids[i])
        return new_input_ids, masked_lm_positions, masked_lm_labels

    def _parse_predictions(self, predictions, all_tokens, output_file=None):
        """
        获取句子级别的困惑度值, 以及疑似错别字位置
        :param predictions: list, loss
        :param all_tokens:
        :param output_file: json file to dump
        :return: list, ppl of each sentence
        """
        result = []
        i = 0
        sentence_loss = 0.0
        word_count_per_sent = 0
        error_ids = []
        sentence = {}
        for word_loss in predictions:
            # start of a sentence
            if all_tokens[i] == "[CLS]":
                sentence = {}
                tokens = []
                sentence_loss = 0.0
                word_count_per_sent = 0
                i += 1
                error_ids = []

            # add token
            prob = float(np.exp(-word_loss[0]))
            tokens.append({"token": all_tokens[i],
                           "prob": prob,
                           "index": word_count_per_sent})
            sentence_loss += word_loss[0]
            word_count_per_sent += 1
            i += 1

            token_count_per_word = 0
            while self.is_subtoken(all_tokens[i]):
                token_count_per_word += 1
                tokens.append({"index": word_count_per_sent,
                               "token": all_tokens[i],
                               "prob": float(np.exp(-word_loss[token_count_per_word]))})
                sentence_loss += word_loss[token_count_per_word]
                i += 1

            # end of a sentence
            if all_tokens[i] == "[SEP]":
                sentence["tokens"] = tokens
                sentence["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
                sentence["error_ids"] = error_ids
                result.append(sentence)
                i += 1

        if output_file:
            tf.logging.info("Saving results to %s" % output_file)
            with tf.gfile.GFile(output_file, "w") as writer:
                writer.write(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def perplexity(self, sentences: list):
        """
        获取句子级别的困惑度值，以及词级别的似然概率
        :param sentences:
        :return: list, ppl of each sentence
        [
            tokens:
                [token:i; prob:0.91,
                 token:love; prob:0.89,
                 token:yo; prob:0.000001
                ],
            ppl:float
        ]
        """
        self.check_bert_detector_initialized()

        predict_examples = self._read_examples(sentences)
        features, all_tokens = self._convert_examples_to_features(predict_examples)

        predict_input_fn = self._input_fn_builder(
            features=features,
            seq_length=self.max_seq_length,
            max_predictions_per_seq=self.max_predictions_per_seq)

        predictions = self.model.predict(input_fn=predict_input_fn)
        result = self._parse_predictions(predictions, all_tokens)
        return result

    def detect(self, sentence):
        """
        句子改错
        :param sentence: 句子文本
        :return: list[list], [error_word, begin_pos, end_pos, error_type]
        """
        sentences = [sentence]
        return self.detect_batch(sentences)[0]

    def detect_batch(self, sentences):
        """
        句子改错
        :param sentences: list, 句子文本
        :return: list[list[list]], [error_word, begin_pos, end_pos, error_type]
        """
        maybe_errors = []
        ppls = self.perplexity(sentences)
        for sentence_result in ppls:
            sentence_err = []
            for token_prob in sentence_result['tokens']:
                token = token_prob['token']
                prob = token_prob['prob']
                index = token_prob['index']
                if prob < self.threshold:
                    item = [token, index, index + 1, ErrorType.char]
                    sentence_err.append(item)
            maybe_errors.append(sentence_err)
        return maybe_errors


if __name__ == "__main__":
    d = BertDetector(bert_model_dir='../data/bert_pytorch/multi_cased_L-12_H-768_A-12',
                     bert_model_vocab='../data/bert_pytorch/multi_cased_L-12_H-768_A-12/vocab.txt')

    error_sentences = ['少先队员因该为老人让座',
                       '少先队员因该为老人让坐',
                       '少 先 队 员 因 该 为老人让座',
                       '少 先 队 员 因 该 为老人让坐',
                       '机七学习是人工智能领遇最能体现智能的一个分支',
                       '机七学习是人工智能领遇最能体现智能的一个分知']
    t1 = time.time()
    err = d.detect_batch(error_sentences)
    t2 = time.time()
    print('batch, time cost:', t2 - t1)
    print("detect sentences:{}".format(err))
    for i in error_sentences:
        err = d.detect(i)
        print("original sentence:{} => detect sentence:{}".format(i, err))
    t3 = time.time()
    print('each, time cost, ', t3 - t2)

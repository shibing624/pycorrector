# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf

import seq2seq_config
from fce_reader import FCEReader
from reader import EOS_ID
from train import create_model


def decode(sess, model, data_reader, data_to_decode,
           corrective_tokens=None, verbose=True):
    """
    Infer the correction sentence
    :param sess:
    :param model:
    :param data_reader:
    :param data_to_decode: an iterable of token lists representing the input
        data we want to decode
    :param corrective_tokens
    :param verbose:
    :return:
    """
    model.batch_size = 1
    corrective_tokens_mask = np.zeros(model.target_vocab_size)
    corrective_tokens_mask[EOS_ID] = 1.0

    if corrective_tokens is None:
        corrective_tokens = set()
    for tokens in corrective_tokens:
        for token in tokens:
            corrective_tokens_mask[data_reader.convert_token_to_id(token)] = 1.0

    for tokens in data_to_decode:
        token_ids = [data_reader.convert_token_to_id(token) for token in tokens]

        # Which bucket does it belong to?
        matching_buckets = [b for b in range(len(model.buckets))
                            if model.buckets[b][0] > len(token_ids)]
        if not matching_buckets:
            # The input string has more tokens than the largest bucket, so we
            # have to skip it.
            continue

        bucket_id = min(matching_buckets)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

        # Get output logits for the sentence.
        _, _, output_logits = model.step(
            sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
            True, corrective_tokens=corrective_tokens_mask)

        oov_input_tokens = [token for token in tokens if
                            data_reader.is_unknown_token(token)]
        outputs = []
        next_oov_token_idx = 0

        for logit in output_logits:
            max_likelihood_token_id = int(np.argmax(logit, axis=1))
            # Check if this logit most likely points to the EOS identifier.
            if max_likelihood_token_id == EOS_ID:
                break

            token = data_reader.convert_id_to_token(max_likelihood_token_id)
            if data_reader.is_unknown_token(token):
                # Replace the "unknown" token with the most probable OOV
                # token from the input.
                if next_oov_token_idx < len(oov_input_tokens):
                    # If we still have OOV input tokens available,
                    # pick the next available one.
                    token = oov_input_tokens[next_oov_token_idx]
                    # Advance to the next OOV input token.
                    next_oov_token_idx += 1
                else:
                    # If we've already used all OOV input tokens,
                    # then we just leave the token as "UNK"
                    pass
            outputs.append(token)
        if verbose:
            decoded_sentence = " ".join(outputs)
            print("Input: {}".format(" ".join(tokens)))
            print("Output: {}\n".format(decoded_sentence))
        yield outputs


def decode_sentence(sess, model, data_reader, sentence, corrective_tokens=set(),
                    verbose=True):
    """Used with InteractiveSession in IPython """
    return next(decode(sess, model, data_reader, [sentence.split()],
                       corrective_tokens=corrective_tokens, verbose=verbose))


def evaluate_accuracy(sess, model, data_reader, corrective_tokens, test_path,
                      max_samples=None):
    """Evaluates the accuracy and BLEU score of the given model."""

    import nltk  # Loading here to avoid having to bundle it in lambda.

    # Build a collection of "baseline" and model-based hypotheses, where the
    # baseline is just the (potentially errant) source sequence.
    baseline_hypotheses = defaultdict(list)  # The model's input
    model_hypotheses = defaultdict(list)  # The actual model's predictions
    targets = defaultdict(list)  # Groundtruth

    errors = []

    n_samples_by_bucket = defaultdict(int)
    n_correct_model_by_bucket = defaultdict(int)
    n_correct_baseline_by_bucket = defaultdict(int)
    n_samples = 0

    # Evaluate the model against all samples in the test data set.
    for source, target in data_reader.read_samples_by_string(test_path):
        matching_buckets = [i for i, bucket in enumerate(model.buckets) if
                            len(source) < bucket[0]]
        if not matching_buckets:
            continue

        bucket_id = matching_buckets[0]

        decoding = next(
            decode(sess, model, data_reader, [source],
                   corrective_tokens=corrective_tokens, verbose=False))
        model_hypotheses[bucket_id].append(decoding)
        if decoding == target:
            n_correct_model_by_bucket[bucket_id] += 1
        else:
            errors.append((decoding, target))

        baseline_hypotheses[bucket_id].append(source)
        if source == target:
            n_correct_baseline_by_bucket[bucket_id] += 1

        # nltk.corpus_bleu expects a list of one or more reference
        # translations per sample, so we wrap the target list in another list
        targets[bucket_id].append([target])

        n_samples_by_bucket[bucket_id] += 1
        n_samples += 1

        if max_samples is not None and n_samples > max_samples:
            break

    # Measure the corpus BLEU score and accuracy for the model and baseline
    # across all buckets.
    for bucket_id in targets.keys():
        baseline_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], baseline_hypotheses[bucket_id])
        model_bleu_score = nltk.translate.bleu_score.corpus_bleu(
            targets[bucket_id], model_hypotheses[bucket_id])
        print("Bucket {}: {}".format(bucket_id, model.buckets[bucket_id]))
        print("\tBaseline BLEU = {:.4f}\n\tModel BLEU = {:.4f}".format(
            baseline_bleu_score, model_bleu_score))
        print("\tBaseline Accuracy: {:.4f}".format(
            1.0 * n_correct_baseline_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))
        print("\tModel Accuracy: {:.4f}".format(
            1.0 * n_correct_model_by_bucket[bucket_id] /
            n_samples_by_bucket[bucket_id]))

    return errors


def main(_):
    print('Correcting error...')
    # Set the model path.
    model_path = seq2seq_config.model_path
    data_reader = FCEReader(seq2seq_config, seq2seq_config.train_path)

    if seq2seq_config.enable_decode_sentence:
        # Correct user's sentences.
        with tf.Session() as session:
            model = create_model(session, True, model_path, config=seq2seq_config)
            print("Enter a sentence you'd like to correct")
            correct_new_sentence = input()
            while correct_new_sentence.lower() != 'no':
                decode_sentence(session, model=model, data_reader=data_reader,
                                sentence=correct_new_sentence,
                                corrective_tokens=data_reader.read_tokens(seq2seq_config.train_path))
                print("Enter a sentence you'd like to correct or press NO")
                correct_new_sentence = input()
    elif seq2seq_config.enable_test_decode:
        # Decode test sentences.
        with tf.Session() as session:
            model = create_model(session, True, model_path, config=seq2seq_config)
            print("Loaded model. Beginning decoding.")
            decodings = decode(session, model=model, data_reader=data_reader,
                               data_to_decode=data_reader.read_tokens(seq2seq_config.test_path),
                               corrective_tokens=data_reader.read_tokens(seq2seq_config.train_path))
            # Write the decoded tokens to stdout.
            for tokens in decodings:
                sys.stdout.flush()


if __name__ == "__main__":
    tf.app.run()

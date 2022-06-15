# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/ThilinaRajapakse/simpletransformers
"""

import os
import pickle
from multiprocessing import Pool

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from pycorrector.utils.logger import logger


def preprocess_batch_for_hf_dataset(dataset, encoder_tokenizer, decoder_tokenizer, args):
    source_inputs = encoder_tokenizer(
        dataset["input_text"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )

    target_inputs = decoder_tokenizer(
        dataset["target_text"],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="np",
        truncation=True,
    )
    source_ids = source_inputs["input_ids"].squeeze()
    target_ids = target_inputs["input_ids"].squeeze()
    src_mask = source_inputs["attention_mask"].squeeze()
    return {
        "input_ids": source_ids,
        "attention_mask": src_mask,
        "decoder_input_ids": target_ids,
    }


def load_hf_dataset(data, encoder_tokenizer, decoder_tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset("csv", data_files=data, delimiter="\t")
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, args=args
        ),
        batched=True,
    )

    column_names = [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
    ]

    dataset.set_format(type="pt", columns=column_names)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    input_text = encoder_tokenizer.encode(
        input_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_text = decoder_tokenizer.encode(
        target_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )
    return torch.flatten(input_text), torch.flatten(target_text)


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name.replace("/", "_") + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(p.imap(preprocess_data, data, chunksize=chunksize), total=len(data), disable=args.silent, )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

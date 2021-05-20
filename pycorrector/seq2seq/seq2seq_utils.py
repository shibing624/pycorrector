# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/ThilinaRajapakse/simpletransformers
"""

import os
import pickle
from multiprocessing import Pool

import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers.models.bart.modeling_bart import shift_tokens_right as _shift_tokens_right

from pycorrector.utils.logger import logger

if transformers.__version__ < "4.2.0":
    shift_tokens_right = lambda input_ids, pad_token_id, decoder_start_token_id: _shift_tokens_right(
        input_ids, pad_token_id)
else:
    shift_tokens_right = _shift_tokens_right


def preprocess_batch_for_hf_dataset(dataset, encoder_tokenizer, decoder_tokenizer, args):
    if args.model_type == "bart":
        input_ids = encoder_tokenizer.batch_encode_plus(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        target_ids = encoder_tokenizer.batch_encode_plus(
            dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
        }
    elif args.model_type == "mbart":
        tokenized_example = encoder_tokenizer.prepare_seq2seq_batch(
            src_texts=dataset["input_text"],
            tgt_texts=dataset["target_text"],
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_length=args.max_seq_length,
            padding="max_length",  # pad_to_max_length=True won't work in this case
            return_tensors="np",
            truncation=True,
        )

        decoder_input_ids = tokenized_example["labels"].clone()
        decoder_input_ids = shift_tokens_right(
            decoder_input_ids, encoder_tokenizer.pad_token_id, encoder_tokenizer.lang_code_to_id[args.tgt_lang]
        )

        labels = tokenized_example["labels"]
        labels[labels == encoder_tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_example["input_ids"].squeeze(),
            "attention_mask": tokenized_example["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids.squeeze(),
            "labels": labels.squeeze(),
        }
    elif args.model_type in ["rag-token", "rag-sequence"]:
        source_inputs = encoder_tokenizer(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        target_inputs = encoder_tokenizer.generator(
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
    else:
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

    if args.model_type == "bart":
        column_names = [
            "source_ids",
            "source_mask",
            "target_ids",
        ]
    elif args.model_type == "mbart":
        column_names = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "labels",
        ]
    else:
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

    if args.model_type in ["rag-token", "rag-sequence"]:
        source_inputs = encoder_tokenizer(
            input_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
        )
        target_inputs = encoder_tokenizer.generator(
            target_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
        )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }
    else:
        input_text = encoder_tokenizer.encode(
            input_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
        )

        target_text = decoder_tokenizer.encode(
            target_text, max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
        )
        return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name.replace("/", "_") + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

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
                logger.info(" Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_data_bart(data):
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }


def preprocess_data_mbart(data):
    input_text, target_text, tokenizer, args = data

    tokenized_example = tokenizer.prepare_seq2seq_batch(
        src_texts=[input_text],
        tgt_texts=[target_text],
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_seq_length,
        padding="max_length",  # pad_to_max_length=True won't work in this case
        return_tensors="pt",
        truncation=True,
    )

    decoder_input_ids = tokenized_example["labels"].clone()
    decoder_input_ids = shift_tokens_right(
        decoder_input_ids, tokenizer.pad_token_id, tokenizer.lang_code_to_id[args.tgt_lang]
    )

    labels = tokenized_example["labels"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_example["input_ids"].squeeze(),
        "attention_mask": tokenized_example["attention_mask"].squeeze(),
        "decoder_input_ids": decoder_input_ids.squeeze(),
        "labels": labels.squeeze(),
    }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
            ]

            preprocess_fn = preprocess_data_mbart if args.model_type == "mbart" else preprocess_data_bart

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(p.imap(preprocess_fn, data, chunksize=chunksize), total=len(data), disable=args.silent, )
                    )
            else:
                self.examples = [preprocess_fn(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def split_text(text, n=100, character=" "):
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents, split_text_n=100, split_text_character=" "):
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text, n=split_text_n, character=split_text_character):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents, ctx_encoder, ctx_tokenizer, device):
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Sequence

import datasets
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index


@dataclass
class GptArgs:
    """
    Model args for a GptModel
    """

    model_class: str = "GptArgs"
    dataset_class: Dataset = None
    learning_rate: float = 2e-5
    manual_seed: int = 42
    fp16: bool = True
    bf16: bool = False
    int8: bool = False
    int4: bool = False
    debug: bool = False
    max_seq_length: int = 256  # max length of input sequence
    max_length: int = 256  # max length of the sequence to be generated
    warmup_steps: int = 50
    report_to = "tensorboard"
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 400
    max_eval_samples: int = 20
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    do_sample: bool = True
    temperature: float = 0.7
    special_tokens_list: list = field(default_factory=list)
    evaluate_during_training: bool = False
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    model_name: str = None
    tokenizer_name: str = None
    reprocess_input_data: bool = False
    silent: bool = False
    no_cache: bool = False
    cache_dir: str = "cache_dir/"
    no_save: bool = False
    save_optimizer_and_scheduler: bool = False
    top_k: float = 40
    top_p: float = 0.9
    model_name_or_path: Optional[str] = field(default="shibing624/chinese-alpaca-plus-7b-hf")
    use_peft: bool = True
    peft_type: str = "LORA"
    peft_bin_name: str = "adapter_model.bin"
    lora_r: int = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["all"]  # ["all"] or ["k_proj"]
    lora_bias = "none"
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    lora_beta: float = 0.85
    num_virtual_tokens: int = 20
    prompt_encoder_hidden_size: int = 128
    num_train_epochs = 3
    max_steps = -1
    per_device_train_batch_size = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps = 1
    save_total_limit = 10
    remove_unused_columns = False
    logging_steps = 50
    resume_from_checkpoint: str = None
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    trust_remote_code: bool = True
    qlora: bool = False
    preprocessing_num_workers: int = 4
    prompt_template_name: str = "vicuna"

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system_prompt: str
    # All messages. format: list of [question, answer]
    messages: Optional[List[Sequence[str]]]
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Conversation prompt
    prompt: str
    # Separator
    sep: str
    # Stop token, default is tokenizer.eos_token
    stop_str: Optional[str] = "</s>"

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
        messages = messages or self.messages
        convs = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        """Append a new message."""
        self.messages.append([query, answer])


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """Register a new conversation template."""
    conv_templates[template.name] = template


"""Vicuna v1.1 template
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT: ",
        sep="</s>",
    )
)

"""Alpaca template"""
register_conv_template(
    Conversation(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.",
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{query}\n\n### Response:\n",
        sep="\n\n",
    )
)

"""Baichuan-13B-Chat template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/f5f47be2adbbdceb784f334d6fa1ca2c73e65097/modeling_baichuan.py#L507
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan-chat",
        system_prompt="",
        messages=[],
        roles=("<reserved_102>", "<reserved_103>"),
        prompt=" <reserved_102> {query} <reserved_103> ",
        sep="</s>",
    )
)

"""ziya template"""
register_conv_template(
    Conversation(
        name="ziya",
        system_prompt="",
        messages=[],
        roles=("<human>", "<bot>"),
        prompt="<human>:{query}\n<bot>:",
        sep="\n",
    )
)

"""Linly template"""
register_conv_template(
    Conversation(
        name="linly",
        system_prompt="",
        messages=[],
        roles=("User", "Bot"),
        prompt="User: {query}\nBot: ",
        sep="\n",
    )
)

"""ChatGLM1 template
source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
"""
register_conv_template(
    Conversation(
        name="chatglm",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n答：",
        sep="\n",
    )
)

"""ChatGLM2 template
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    # source:
    Conversation(
        name="chatglm2",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n\n答：",
        sep="\n\n",
    )
)

"""ChatGLM3 template
Support: https://huggingface.co/THUDM/chatglm3-6b
source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
"""
register_conv_template(
    Conversation(
        name="chatglm3",
        system_prompt="",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|assistant|>",
        sep="\n",
        stop_str="<|user|>",
    )
)

"""Phoenix template"""
register_conv_template(
    Conversation(
        name="phoenix",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: <s>{query}</s>Assistant: ",
        sep="</s>",
    )
)

"""belle template
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_conv_template(
    Conversation(
        name="belle",
        system_prompt="",
        messages=[],
        roles=("Human", "Belle"),
        prompt="Human: {query}\n\nBelle: ",
        sep="\n\n",
    )
)

"""aquila template
Supports: https://huggingface.co/qhduan/aquilachat-7b
"""
register_conv_template(
    Conversation(
        name="aquila",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}###Assistant: ",
        sep="###",
    )
)

"""intern template
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_conv_template(
    Conversation(
        name="intern",
        system_prompt="",
        messages=[],
        roles=("<|User|>", "<|Bot|>"),
        prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
        sep="<eoa>\n",
        stop_str="<eoa>",
    )
)

"""StarChat template"""
register_conv_template(
    Conversation(
        name="starchat",
        system_prompt="<system>\n",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
        sep="<|end|>\n",
        stop_str="<|end|>",
    )
)

"""llama2 template
reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
"""
register_conv_template(
    Conversation(
        name="llama2",
        system_prompt="<<SYS>>\nYou are a helpful, respectful and honest assistant. "
                      "Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, "
                      "toxic, dangerous, or illegal content. "
                      "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                      "If a question does not make any sense, or is not factually coherent, "
                      "explain why instead of answering something not correct. "
                      "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST] ",
        sep="</s>",
    )
)

"""llama2-zh template
Sources: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_conv_template(
    Conversation(
        name="llama2-zh",
        system_prompt="[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST]",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST] ",
        sep="</s>",
    )
)

"""XVERSE template
Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_conv_template(
    Conversation(
        name="xverse",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: ",
        sep="</s>",
    )
)

"""Qwen template
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
"""
register_conv_template(
    Conversation(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name]


def preprocess_function(examples, tokenizer, args):
    """
    Preprocessing the datasets.
        part of code modified from https://github.com/lm-sys/FastChat
    """
    input_ids_list = []
    targets_list = []
    roles = ["human", "gpt"]
    prompt_template = get_conv_template(args.prompt_template_name)
    max_source_length = args.max_seq_length
    max_target_length = args.max_length
    max_full_length = max_source_length + max_target_length

    def get_dialog(examples):
        for i, source in enumerate(examples['conversations']):
            if len(source) < 2:
                continue
            data_role = source[0].get("from", "")
            if data_role not in roles or data_role != roles[0]:
                # Skip the first one if it is not from human
                break
            messages = []
            for j, sentence in enumerate(source):
                data_role = sentence.get("from", "")
                if data_role not in roles:
                    logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                    break
                if data_role == roles[j % 2]:
                    messages.append(sentence["value"])
            if len(messages) % 2 != 0:
                continue
            # Convert the list to pairs of elements
            history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
            yield prompt_template.get_dialog(history_messages)

    for dialog in get_dialog(examples):
        input_ids, labels = [], []

        for i in range(len(dialog) // 2):
            source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
            target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length - 1:  # eos token
                target_ids = target_ids[:max_target_length - 1]
            if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                source_ids = source_ids[1:]
            if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                target_ids = target_ids[:-1]
            if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_full_length:
                break

            input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        input_ids_list.append(input_ids)
        targets_list.append(labels)

    return dict(
        input_ids=input_ids_list,
        labels=targets_list,
    )


def filter_empty_labels(example):
    """Remove empty labels dataset."""
    return not all(label == IGNORE_INDEX for label in example["labels"])


def load_supervised_dataset(tokenizer, args, data, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = datasets.load_from_disk(data)
        else:
            dataset = load_dataset(
                data,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset['train']
        if mode == 'dev' and args.max_eval_samples is not None:
            max_eval_samples = min(len(dataset), args.max_eval_samples)
            dataset = dataset.select(range(max_eval_samples))
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.shuffle().map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, args=args),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    dataset = dataset.filter(filter_empty_labels, num_proc=args.preprocessing_num_workers)

    return dataset


class GptSupervisedDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)

            self.examples = load_supervised_dataset(tokenizer, args, data, mode)
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

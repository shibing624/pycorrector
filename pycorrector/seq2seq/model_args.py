# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/ThilinaRajapakse/simpletransformers
"""
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count

from torch.utils.data import Dataset


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count


def get_special_tokens():
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass
class ModelArgs:
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    cosine_schedule_num_cycles: float = 0.5
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    dataloader_num_workers: int = 0
    do_lower_case: bool = False
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = None
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-5
    local_rank: int = -1
    logging_steps: int = 50
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    model_name: str = None
    model_type: str = None
    multiprocessing_chunksize: int = -1
    n_gpu: int = 1
    no_cache: bool = False
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = 1
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = False
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    process_count: int = field(default_factory=get_default_process_count)
    quantized_model: bool = False
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = False
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 2000
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = None
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_batch_size: int = 8
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = True
    use_multiprocessing_for_evaluation: bool = True
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["tokenizer_type"] is not None and not isinstance(args_dict["tokenizer_type"], str):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)



@dataclass
class Seq2SeqArgs(ModelArgs):
    """
    Model args for a Seq2SeqModel
    """

    model_class: str = "Seq2SeqModel"
    base_marian_model_name: str = None
    dataset_class: Dataset = None
    do_sample: bool = False
    early_stopping: bool = True
    evaluate_generated_text: bool = False
    faiss_d: int = 768
    faiss_m: int = 128
    length_penalty: float = 2.0
    max_length: int = 20
    max_steps: int = -1
    num_beams: int = 1
    num_return_sequences: int = 1
    rag_embed_batch_size: int = 16
    repetition_penalty: float = 1.0
    top_k: float = None
    top_p: float = None
    use_multiprocessed_decoding: bool = False
    save_knowledge_dataset: bool = True
    save_knowledge_dataset_with_checkpoints: bool = False
    split_text_character: str = " "
    split_text_n: int = 100
    src_lang: str = "en_XX"
    tgt_lang: str = "ro_RO"

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["dataset_class"] is not None:
                args_dict["dataset_class"] = type(args_dict["dataset_class"]).__name__
            json.dump(self.get_args_for_saving(), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)
                if model_args["dataset_class"]:
                    warnings.warn(
                        "This model was trained using a custom dataset_class."
                        "This cannot be loaded automatically and must be specified in the model args"
                        "when loading the model."
                    )
                self.update_from_dict(model_args)


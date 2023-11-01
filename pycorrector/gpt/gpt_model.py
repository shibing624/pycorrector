# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi round conversation SFT model
"""
import math
import os
import random
from threading import Thread
from typing import List, Tuple, Optional

import numpy as np
import torch
from loguru import logger
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    BloomTokenizerFast,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TextIteratorStreamer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME

from pycorrector.gpt.gpt_utils import GptSupervisedDataset, IGNORE_INDEX, GptArgs, get_conv_template

has_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


class GptModel:
    def __init__(
            self,
            model_type,
            model_name,
            peft_name: Optional[str] = None,
            args: Optional[dict] = None,
            use_cuda: Optional[bool] = has_cuda,
            cuda_device: Optional[int] = -1,
            **kwargs,
    ):

        """
        Initializes a GptModel model.

        Args:
            model_type: The type of model (llama, bloom, baichuan, auto)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            peft_name (optional): Peft model name
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (int, optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"
        model_type = model_type.lower()
        self.args = GptArgs()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, GptArgs):
            self.args = args

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if torch.cuda.is_available() > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.device_map = "auto"
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
                    self.device_map = {"": int(cuda_device)}
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_map = {"": "mps"}
            else:
                self.device = "cpu"
                self.device_map = {"": "cpu"}
        logger.debug(f"Device: {self.device}")
        if not use_cuda:
            self.args.fp16 = False
            self.args.int8 = False
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp = self.world_size != 1
        if self.ddp:
            self.device_map = {"": self.local_rank}

        self.results = {}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if model_name is None:
            model_name = self.args.model_name_or_path

        if self.args.bf16:
            self.args.fp16 = False
        if self.args.fp16:
            self.args.bf16 = False
        self.torch_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        self.config = config_class.from_pretrained(
            model_name,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=self.torch_dtype,
            **kwargs
        )
        self.model = model_class.from_pretrained(
            model_name,
            config=self.config,
            load_in_8bit=self.args.int8,
            load_in_4bit=self.args.int4,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=self.device_map,
            trust_remote_code=self.args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=self.args.int4,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            ) if self.args.qlora else None,
        )

        self.tokenizer_class = tokenizer_class
        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.args.tokenizer_name, trust_remote_code=self.args.trust_remote_code)
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, trust_remote_code=self.args.trust_remote_code)
            self.args.tokenizer_name = self.args.model_name
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "</s>"  # eos token is required for SFT
            logger.debug("Add eos token: {}".format(self.tokenizer.eos_token))
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Add pad token: {}".format(self.tokenizer.pad_token))

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "Llama_from_scratch"
        else:
            self.args.model_name = model_name

        self.peft_name = peft_name
        if self.args.use_peft and self.peft_name:
            self.load_peft_model()

    def load_peft_model(self):
        """Load peft model"""
        self.model = PeftModel.from_pretrained(
            self.model,
            self.peft_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model = self.model.merge_and_unload()
        logger.info(f"Loaded peft model from {self.peft_name}")

    def find_all_linear_names(self, int4=False, int8=False):
        cls = torch.nn.Linear
        if int4 or int8:
            import bitsandbytes as bnb
            if int4:
                cls = bnb.nn.Linear4bit
            elif int8:
                cls = bnb.nn.Linear8bitLt
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                # last layer is not add to lora_module_names
                if 'lm_head' in name:
                    continue
                if 'output_layer' in name:
                    continue
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)

    def train_model(
            self,
            train_data,
            output_dir=None,
            args=None,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: json file path or Pandas DataFrame containing 1 columns - `conversations`.
                format: {"conversations":[{"from":"human","value":"Mike的妈妈有4个孩子; 其中3个是 Luis、Drake 和 Matilda。 第4个孩子叫什么？"},{"from":"gpt","value":"Mike。"}]}
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            verbose (optional): If True, all of the warnings related to data processing will be printed. 
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"
        if args:
            self.args.update_from_dict(args)
        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir
        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        # Setup train args
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            gradient_checkpointing=self.args.gradient_checkpointing,
            torch_compile=self.args.torch_compile,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            optim=self.args.optimizer,
            save_strategy=self.args.save_strategy,
            evaluation_strategy='steps' if eval_data is not None else 'no',
            eval_steps=self.args.eval_steps if eval_data is not None else None,
            load_best_model_at_end=True if eval_data is not None else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            remove_unused_columns=self.args.remove_unused_columns,
            report_to=self.args.report_to,
            overwrite_output_dir=self.args.overwrite_output_dir,
            no_cuda=True if self.device == "cpu" else False,
            **kwargs
        )
        resume_from_checkpoint = self.args.resume_from_checkpoint
        if self.args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
        if 'all' in self.args.lora_target_modules:
            self.args.lora_target_modules = self.find_all_linear_names(self.args.int4, self.args.int8)
        # setup peft
        if self.args.use_peft:
            if self.args.int8 or self.args.int4:
                self.model = prepare_model_for_kbit_training(self.model, self.args.gradient_checkpointing)

            peft_type = self.args.peft_type.upper()
            logger.info(f"Using PEFT type: {peft_type}")
            # add peft config
            if peft_type == 'LORA':
                logger.debug(f"Using list modules for LoRA: {self.args.lora_target_modules}")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )
            elif peft_type == 'ADALORA':
                from peft import AdaLoraConfig
                logger.debug(f"Using list modules for LoRA: {self.args.lora_target_modules}")
                peft_config = AdaLoraConfig(
                    init_r=self.args.adalora_init_r,
                    r=self.args.lora_r,
                    beta1=self.args.lora_beta,
                    beta2=self.args.lora_beta,
                    tinit=self.args.adalora_tinit,
                    tfinal=self.args.adalora_tfinal,
                    deltaT=self.args.adalora_delta_t,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                )
            elif peft_type == 'PROMPT_TUNING':
                from peft import PromptTuningConfig

                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                )
            elif peft_type == 'P_TUNING':
                from peft import PromptEncoderConfig

                peft_config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size
                )
            elif peft_type == 'PREFIX_TUNING':
                from peft import PrefixTuningConfig

                peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size,
                    prefix_projection=True,
                )
                self.model.gradient_checkpointing_disable()
            else:
                logger.warning(f"Wrong type of peft. Set to default lora")
                logger.debug(f"Using list modules for LoRA: {self.args.lora_target_modules}")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )

            if isinstance(self.model, PeftModel):
                logger.debug("Merge peft weights to base model")
                self.model = self.model.merge_and_unload()
            self.model = get_peft_model(self.model, peft_config)

            if resume_from_checkpoint:
                # Check the available weights and load them
                checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
                if not os.path.exists(checkpoint_name):
                    checkpoint_name = os.path.join(
                        resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
                    resume_from_checkpoint = (
                        False  # So the trainer won't try loading its state
                    )
                # The two files above have a different name depending on how they were saved, but are actually the same.
                if os.path.exists(checkpoint_name):
                    logger.info(f"Restarting from {checkpoint_name}")
                    adapters_weights = torch.load(checkpoint_name, map_location='cpu')
                    set_peft_model_state_dict(self.model, adapters_weights)
                else:
                    logger.warning(f"Checkpoint {checkpoint_name} not found")
                    resume_from_checkpoint = None

            self.model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        else:
            logger.info("Fine-tuning method: Full parameters training")
            # self.model = self.model.float()
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Tokenizer: {self.tokenizer}")

        # load dataset
        train_dataset = self.load_and_cache_examples(train_data)
        if verbose:
            logger.debug(f"train_dataset len: {len(train_dataset)}, train_dataset[0]: {train_dataset[0]}")
            logger.debug("Tokenized training example:")
            logger.debug(f"Decode input_ids[0]: {self.tokenizer.decode(train_dataset[0]['input_ids'])}")
            replaced_labels = [label if label != IGNORE_INDEX else self.tokenizer.pad_token_id
                               for label in list(train_dataset[0]['labels'])]
            logger.debug(f"Decode labels[0]: {self.tokenizer.decode(replaced_labels)}")
        eval_dataset = None
        if eval_data is not None:
            eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)
            if verbose:
                logger.debug(f"eval_dataset len: {len(eval_dataset)}, eval_dataset[0]: {eval_dataset[0]}")

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        if training_args.local_rank <= 0:
            logger.info(f"Training/evaluation parameters {training_args}")

        # Update model train config
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
        self.model.enable_input_require_grads()
        if not self.ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        # Initialize our Trainer
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX)
        trainer = SavePeftModelTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_data is not None else None,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Training
        logger.info("*** Train ***")
        sample = next(iter(trainer.get_train_dataloader()))
        logger.debug(f"Train dataloader example: {sample}")
        logger.debug(f"Detail input_ids: {sample['input_ids'][:3]}, \nlabels: {sample['labels'][:3]}")
        logger.debug(f"Decode input_ids[0]: {self.tokenizer.decode(sample['input_ids'][0])}")
        replaced_labels = [label if label != IGNORE_INDEX else
                           self.tokenizer.pad_token_id for label in sample['labels'][0]]
        logger.debug(f"Decode labels[0]: {self.tokenizer.decode(replaced_labels)}")

        (global_step, training_loss, metrics) = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.results.update(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        self.model.config.use_cache = True  # enable cache after training
        trainer.save_state()
        self.save_model(model=self.model)

        if eval_data is not None:
            logger.info("*** Evaluate ***")
            if self.args.fp16:
                self.model.half()
            metrics = trainer.evaluate(metric_key_prefix="eval")
            metrics['eval_samples'] = len(eval_dataset)
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            logger.debug(f"eval metrics: {metrics}")
            self.results.update(metrics)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if verbose and training_args.local_rank <= 0:
            logger.debug(f"metrics: {self.results}")
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )
        return global_step, training_loss

    @torch.inference_mode()
    def predict(
            self,
            sentences: List[str],
            skip_prompt: bool = True,
            prompt_template_name: str = 'vicuna',
            max_length: int = None,
            do_sample: bool = None,
            temperature: float = None,
            repetition_penalty: float = None,
            eval_batch_size: int = None,
            **kwargs
    ) -> List[str]:
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
            skip_prompt: Whether to skip the prompt when generating text.
            prompt_template_name: The name of the prompt template to use.
            max_length: The maximum length of the generated text.
            do_sample: Whether or not to use sampling ; use greedy decoding otherwise.
            temperature: The value used to module the next token probabilities.
            repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty.
            eval_batch_size: Batch size to use for evaluation.
            **kwargs: Additional arguments for generating sequences.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self.model.eval()
        if self.args.fp16:
            self.model.half()
        prompt_template = get_conv_template(prompt_template_name or self.args.prompt_template_name)
        if not eval_batch_size:
            eval_batch_size = self.args.eval_batch_size

        all_outputs = []
        # Batching
        for batch in tqdm(
                [
                    sentences[i: i + eval_batch_size]
                    for i in range(0, len(sentences), eval_batch_size)
                ],
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            if prompt_template_name:
                batch = [prompt_template.get_prompt(messages=[[s, '']]) for s in batch]
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            generation_kwargs = dict(
                max_new_tokens=max_length if max_length is not None else self.args.max_length,
                do_sample=do_sample if do_sample is not None else self.args.do_sample,
                temperature=temperature if temperature is not None else self.args.temperature,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.args.repetition_penalty,
            )
            outputs = self.model.generate(
                input_ids=input_ids,
                **generation_kwargs,
                **kwargs,
            )
            for prompt, generated_sequence in zip(batch, outputs):
                # Decode text
                prompt_len = len(input_ids[0])
                generated_sequence = generated_sequence[prompt_len:]
                gen_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                stop_str = self.tokenizer.eos_token or prompt_template.stop_str
                pos = gen_text.find(stop_str)
                if pos != -1:
                    gen_text = gen_text[:pos]
                if not skip_prompt:
                    gen_text = prompt + gen_text
                all_outputs.append(gen_text)

        return all_outputs

    @torch.inference_mode()
    def chat(
            self,
            query: str,
            history: List[Tuple[str, str]] = None,
            skip_prompt: bool = True,
            prompt_template_name: str = "vicuna",
            max_new_tokens: int = None,
            do_sample: bool = None,
            temperature: float = None,
            repetition_penalty: float = None,
            context_len: int = 2048,
            **kwargs
    ):
        """Chat model with multi turn conversation."""
        prompt_template = get_conv_template(prompt_template_name or self.args.prompt_template_name)

        if history is None:
            history = []
        history.append([query, ''])
        prompt = prompt_template.get_prompt(messages=history)
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=60.0, skip_prompt=skip_prompt, skip_special_tokens=True)
        input_ids = self.tokenizer(prompt).input_ids
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.args.max_length
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample if do_sample is not None else self.args.do_sample,
            temperature=temperature if temperature is not None else self.args.temperature,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.args.repetition_penalty,
            streamer=streamer,
            **kwargs,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        stop_str = self.tokenizer.eos_token or prompt_template.stop_str
        generated_text = ""
        for new_text in streamer:
            stop = False
            pos = new_text.find(stop_str)
            if pos != -1:
                new_text = new_text[:pos]
                stop = True
            generated_text += new_text
            if stop:
                break
        response = generated_text.strip()
        history = history + [[query, response]]
        return response, history

    def load_and_cache_examples(
            self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a LlamaDataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, data, mode)
        else:
            return GptSupervisedDataset(tokenizer, args, data, mode)

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        """Save the model and the tokenizer."""
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

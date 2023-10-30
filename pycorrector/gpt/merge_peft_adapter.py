# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Usage:
python merge_peft_adapter.py \
    --model_type llama \
    --base_model_name_or_path path/to/llama/model \
    --tokenizer_path path/to/llama/tokenizer \
    --peft_model_path path/to/lora/model \
    --output_dir merged

after merged, chatglm and baichuan model need copy python script to output dir.
"""

import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model_name_or_path', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Please specify tokenization path.")
    parser.add_argument('--peft_model_path', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--output_dir', default='./merged', type=str)
    args = parser.parse_args()
    print(args)

    base_model_path = args.base_model_name_or_path
    peft_model_path = args.peft_model_path
    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {peft_model_path}")
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    base_model = model_class.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    if args.tokenizer_path:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = tokenizer_class.from_pretrained(peft_model_path, trust_remote_code=True)
    if args.resize_emb:
        base_model_token_size = base_model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = lora_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir)
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()

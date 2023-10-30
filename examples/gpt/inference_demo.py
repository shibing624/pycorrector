# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use deepspeed to inference with multi-gpus

usage:
CUDA_VISIBLE_DEVICES=0 python inference_demo.py --model_type baichuan --base_model shibing624/vicuna-baichuan-13b-chat --interactive
"""
import argparse
import json
import sys

sys.path.append('../..')
from pycorrector.gpt.gpt_model import GptModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='baichuan', type=str)
    parser.add_argument('--base_model', default='shibing624/vicuna-baichuan-13b-chat', type=str)
    parser.add_argument('--lora_model', default="", type=str, help="If not set, perform inference on the base model")
    parser.add_argument('--prompt_template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan-chat, chatglm2 etc.")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode")
    parser.add_argument('--single_round', action='store_true',
                        help="Whether to generate single round dialogue, default is multi-round dialogue")
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--predictions_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    args = parser.parse_args()
    print(args)

    model = GptModel(args.model_type, args.base_model, peft_name=args.lora_model)
    # test data
    if args.data_file is None:
        examples = [
            "介绍下北京",
            "乙肝和丙肝的区别？",
            "失眠怎么办？",
            '用一句话描述地球为什么是独一无二的。',
            "Tell me about alpacas.",
            "Tell me about the president of Mexico in 2019.",
        ]
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    if args.interactive:
        print(f"Start inference with interactive mode. enable multi round: {not args.single_round}")
        history = []
        while True:
            raw_input_text = input("Input:")
            if len(raw_input_text.strip()) == 0:
                break
            if args.single_round:
                response = model.predict([raw_input_text], prompt_template_name=args.prompt_template_name)[0]
            else:
                response, history = model.chat(
                    raw_input_text, history=history, prompt_template_name=args.prompt_template_name)
            print("Response: ", response)
            print("\n")
    else:
        print("Start inference.")
        results = []
        responses = model.predict(
            examples,
            prompt_template_name=args.prompt_template_name,
            eval_batch_size=args.batch_size
        )
        for index, example, response in zip(range(len(examples)), examples, responses):
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")
            results.append({"Input": example, "Output": response})
        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f'save to {args.predictions_file}, size: {len(results)}')


if __name__ == '__main__':
    main()

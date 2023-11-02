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
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
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
        print(f"Start inference with interactive mode.")
        history = []
        while True:
            try:
                query = input("Input:")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "clear":
                history = []
                print("history cleared.")
                continue
            if query.strip() == 'exit':
                break
            print("Response:", end='', flush=True)
            try:
                response = ""
                for new_token in model.chat(
                        query,
                        history=history,
                        prompt_template_name=args.prompt_template_name,
                        stream=True
                ):
                    print(new_token, end='', flush=True)
                    response += new_token
                history = history + [[query, response]]
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, stop.")
                continue
            print()
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
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f'save to {args.output_file}, size: {len(results)}')


if __name__ == '__main__':
    main()

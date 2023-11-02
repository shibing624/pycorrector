# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys

from loguru import logger

sys.path.append('../..')
from pycorrector.gpt.gpt_model import GptModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/grammar/train_sharegpt.jsonl', type=str, help='Train file')
    parser.add_argument('--test_file', default='../data/grammar/test_sharegpt.jsonl', type=str, help='Test file')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--bf16', action='store_true', help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--output_dir', default='./outputs-chatglm-demo/', type=str, help='Model output directory')
    parser.add_argument('--prompt_template_name', default='vicuna', type=str, help='Prompt template name')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=128, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=50, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=50, type=int, help='Save checkpoint every X steps')
    parser.add_argument("--local_rank", type=int, help="Used by dist launchers")
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "use_peft": True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "bf16": args.bf16,
            "prompt_template_name": args.prompt_template_name,
        }
        model = GptModel(args.model_type, args.model_name, args=model_args)
        model.train_model(args.train_file, eval_data=args.test_file)
    if args.do_predict:
        if model is None:
            model = GptModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )
        prefix = "对下面的文本纠错\n\n"
        sents = [
            "美国总统特朗普访日，不仅吸引了美日民众的关注，中国人民也同样密切关注。",
            "这块名表带带相传",
            "少先队员因该为老人让坐",
        ]
        response = model.predict([prefix + i for i in sents])
        print(response)

        # Chat model with multi turns conversation
        response, history = model.chat("简单介绍下北京", history=None)
        print(response, history)
        response, history = model.chat('继续', history=history)
        print(response)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio>=3.50.2
"""
import argparse
import sys
from threading import Thread

import gradio as gr
import torch
from transformers import TextIteratorStreamer

sys.path.append("../..")
from pycorrector.gpt.gpt_model import GptModel
from pycorrector.gpt.gpt_utils import get_conv_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan2, chatglm2 etc.")
    parser.add_argument('--share', action='store_true', help='Share gradio')
    parser.add_argument('--port', default=8081, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)

    m = GptModel(args.model_type, args.base_model, peft_name=args.lora_model)

    prompt_template = get_conv_template(args.template_name)
    stop_str = m.tokenizer.eos_token or prompt_template.stop_str

    def predict(message, history):
        """Generate answer from prompt with GPT and stream the output"""
        history_messages = history + [[message, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages)
        streamer = TextIteratorStreamer(m.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = m.tokenizer(prompt).input_ids
        context_len = 2048
        max_new_tokens = 512
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(m.device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.0,
        )
        thread = Thread(target=m.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask me question", lines=4, scale=9),
        title="pycorrector: text correction use gpt model",
        description="github: [shibing624/pycorrector](https://github.com/shibing624/pycorrector)",
        theme="soft",
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()

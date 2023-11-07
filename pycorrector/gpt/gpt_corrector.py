# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import time
from typing import List
from typing import Optional

from loguru import logger

sys.path.append('../..')
from pycorrector.utils.tokenizer import split_text_into_sentences_by_length
from pycorrector.gpt.gpt_model import GptModel
from pycorrector.utils.error_utils import get_errors_for_diff_length


class GptCorrector(GptModel):
    def __init__(
            self,
            model_name_or_path: str = "THUDM/chatglm3-6b",
            model_type: str = 'chatglm',
            peft_name: Optional[str] = "shibing624/chatglm3-6b-csc-chinese-lora",
            **kwargs,
    ):
        t1 = time.time()
        super(GptCorrector, self).__init__(
            model_type=model_type,
            model_name=model_name_or_path,
            peft_name=peft_name,
            **kwargs,
        )
        logger.debug('Loaded gpt csc model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    def correct_batch(
            self,
            sentences: List[str],
            max_length: int = 512,
            batch_size: int = 12,
            prompt_template_name: str = 'vicuna',
            prefix_prompt: str = None,
            **kwargs
    ):
        """
        Correct sentences with gpt model.
        :param sentences: list, input sentences
        :param max_length: int, max length of input sentence
        :param batch_size: int, batch size
        :param prompt_template_name: str, prompt template name
        :param prefix_prompt: str, prefix of prompt
        :param kwargs: dict, other params
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        if prefix_prompt is None:
            prefix_prompt = "对下面的文本纠错\n\n"
        input_sents = []
        sent_map = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) > max_length:
                # split long sentence into short ones
                short_sentences = [i[0] for i in split_text_into_sentences_by_length(sentence, max_length)]
                input_sents.extend(short_sentences)
                sent_map.extend([idx] * len(short_sentences))
            else:
                input_sents.append(sentence)
                sent_map.append(idx)
        input_sents = [prefix_prompt + s for s in input_sents]
        # predict all sentences
        corrected_sents = self.predict(
            input_sents,
            max_length=max_length,
            prompt_template_name=prompt_template_name,
            eval_batch_size=batch_size,
            **kwargs
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, corrected_sents):
            corrected_sentences[idx] += corrected_sent

        new_corrected_sentences = []
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            new_corrected_sent, sub_details = get_errors_for_diff_length(corrected_sent, sentences[idx])
            new_corrected_sentences.append(new_corrected_sent)
            corrected_details.append(sub_details)
        return [{'source': s, 'target': c, 'errors': e} for s, c, e in
                zip(sentences, new_corrected_sentences, corrected_details)]

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with gpt csc model"""
        return self.correct_batch([sentence], **kwargs)[0]

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


class GptCorrector(GptModel):
    def __init__(
            self,
            model_name_or_path: str = "shibing624/chatglm3-6b-csc-chinese-merged",
            model_type: str = 'chatglm',
            peft_name: Optional[str] = None,
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
            batch_size: int = 4,
            prompt_template_name: str = 'vicuna',
            **kwargs
    ):
        """
        Correct sentences with gpt model.
        :param sentences: list, input sentences
        :param max_length: int, max length of input sentence
        :param batch_size: int, batch size
        :param prompt_template_name: str, prompt template name
        :param kwargs: dict, other params
        """
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

        return corrected_sentences

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with gpt csc model"""
        return self.correct_batch([sentence], **kwargs)[0]


if __name__ == "__main__":
    m = GptCorrector()
    error_sentences = [
        '内容提要——在知识产权学科领域里',
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '老是较书。',
        '遇到一位很棒的奴生跟我聊天。',
        '他的语说的很好，法语也不错',
        '他法语说的很好，的语也不错',
        '他们的吵翻很不错，再说他们做的咖喱鸡也好吃',
        '影像小孩子想的快，学习管理的斑法',
    ]
    t2 = time.time()
    res = m.correct_batch(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} err:{}".format(sent, r[0], r[1]))
    print('[batch]spend time:', time.time() - t2)

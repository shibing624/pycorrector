# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from paddlenlp import Taskflow


class ErnieCscCorrector:
    def __init__(self, model_name_or_path="ernie-csc"):
        self.text_correction = Taskflow("text_correction", model=model_name_or_path)

    def correct(self, sentence: str):
        """
        句子纠错
        :param sentence: 句子
        :return: list[{'source': wrong_text,
                   'target': right_text,
                   'errors': [{'position': pos, 'correction': {wrong_word: correct_word}}]}]
        """
        return self.text_correction(sentence)

    def correct_batch(self, sentences: List[str]):
        """
        批量句子纠错
        :param sentences: 句子文本列表
        :return: list[{'source': wrong_text,
                   'target': right_text,
                   'errors': [{'position': pos, 'correction': {wrong_word: correct_word}}]}]
        """
        return [self.text_correction(text) for text in sentences]

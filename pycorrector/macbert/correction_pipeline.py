# -*- coding: utf-8 -*-
"""
@Time   :   2021-02-03 18:22:45
@File   :   correction_pipeline.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com

reference: https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/text2text_generation.py.

"""
import sys
from collections import OrderedDict

import torch

sys.path.append('../..')

from pycorrector.transformers import Pipeline, BertConfig, BertForMaskedLM
from pycorrector.transformers.tokenization_utils import TruncationStrategy

MODEL_MAC_BERT_LM_MAPPING = OrderedDict(
    [
        # Model for MacBert LM
        (BertConfig, BertForMaskedLM)
    ]
)


class CorrectionPipeline(Pipeline):
    """
    因transformers没有内置的较合适的Pipeline，故新建了一个Pipeline类。
    """
    return_name = "corrected"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(
            MODEL_MAC_BERT_LM_MAPPING
        )

    @staticmethod
    def check_inputs(input_length: int, max_length: int):
        """
        Checks wether there might be something wrong with given input with regard to the model.
        """
        if input_length > max_length:
            raise ValueError(f"max length of input text need less than {max_length}.")
        return True

    def __call__(
            self,
            *args,
            return_tensors=False,
            return_text=True,
            clean_up_tokenization_spaces=False,
            truncation=TruncationStrategy.DO_NOT_TRUNCATE,
            **correct_kwargs
    ):
        r"""
        Generate the output text(s) using text(s) given as inputs.
        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Input text for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).
        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:
            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            assert (self.tokenizer.pad_token_id is not None
                    ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                " `args[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    args[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, padding=padding, truncation=truncation)

            inputs = self.ensure_tensor_on_device(**inputs)
            input_length = inputs["input_ids"].shape[-1]

            max_length = self.model.config.max_position_embeddings
            self.check_inputs(input_length, max_length)

            corrections = self.model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **correct_kwargs,
            ).logits
            corrections = torch.argmax(corrections, dim=-1)
            results = []
            for correction in corrections:
                record = {}
                if return_tensors:
                    record[f"{self.return_name}_token_ids"] = correction
                if return_text:
                    record[f"{self.return_name}_text"] = self.tokenizer.decode(
                        correction,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    ).replace(' ', '')
                results.append(record)
            return results

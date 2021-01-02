# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
import csv
import json
import os
import pickle
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from os.path import abspath, exists
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pycorrector.utils.logger import logger
from .configuration_utils import PretrainedConfig
from .file_utils import add_end_docstrings, is_tf_available, is_torch_available
from .modelcard import ModelCard
from .models.auto.configuration_auto import AutoConfig
from .models.auto.tokenization_auto import AutoTokenizer
from .models.bert.tokenization_bert import BasicTokenizer
from .tokenization_utils import PreTrainedTokenizer

if is_torch_available():
    import torch

    from .models.auto.modeling_auto import (
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )


def get_framework(model, revision: Optional[str] = None):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
    if not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    if isinstance(model, str):
        if is_torch_available() and not is_tf_available():
            model = AutoModel.from_pretrained(model, revision=revision)
        else:
            try:
                model = AutoModel.from_pretrained(model, revision=revision)
            except OSError:
                model = None

    framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"
    return framework


def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -> str:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (:obj:`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (:obj:`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (:obj:`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        :obj:`str` The model string representing the default model for this pipeline
    """
    if is_torch_available() and not is_tf_available():
        framework = "pt"
    elif is_tf_available() and not is_torch_available():
        framework = "tf"

    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError("The task does not provide any default models for options {}".format(task_options))
        default_models = defaults[task_options]["model"]
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # XXX This error message needs to be updated to be more generic if more tasks are going to become
        # parametrized
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    if framework is None:
        framework = "pt"

    return default_models[framework]


class PipelineException(Exception):
    """
    Raised by a :class:`~transformers.Pipeline` when handling __call__.

    Args:
        task (:obj:`str`): The task of the pipeline.
        model (:obj:`str`): The model used by the pipeline.
        reason (:obj:`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each :class:`~transformers.pipelines.Pipeline`.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    :obj:`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets
    columns to pipelines keyword arguments through the :obj:`dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
            self,
            output_path: Optional[str],
            input_path: Optional[str],
            column: Optional[str],
            overwrite: bool = False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError("{} already exists on disk".format(self.output_path))

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError("{} doesnt exist on disk".format(self.input_path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.

        Returns:
            :obj:`str`: Path where the data has been saved.
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
            format: str,
            output_path: Optional[str],
            input_path: Optional[str],
            column: Optional[str],
            overwrite=False,
    ) -> "PipelineDataFormat":
        """
        Creates an instance of the right subclass of :class:`~transformers.pipelines.PipelineDataFormat` depending on
        :obj:`format`.

        Args:
            format: (:obj:`str`):
                The format of the desired pipeline. Acceptable values are :obj:`"json"`, :obj:`"csv"` or :obj:`"pipe"`.
            output_path (:obj:`str`, `optional`):
                Where to save the outgoing data.
            input_path (:obj:`str`, `optional`):
                Where to look for the input data.
            column (:obj:`str`, `optional`):
                The column to read.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to overwrite the :obj:`output_path`.

        Returns:
            :class:`~transformers.pipelines.PipelineDataFormat`: The proper data format.
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError("Unknown reader {} (Available reader are json/csv/pipe)".format(format))


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __init__(
            self,
            output_path: Optional[str],
            input_path: Optional[str],
            column: Optional[str],
            overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __init__(
            self,
            output_path: Optional[str],
            input_path: Optional[str],
            column: Optional[str],
            overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (:obj:`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:

                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (:obj:`dict`): The data to store.
        """
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


PIPELINE_INIT_ARGS = r"""
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """

    default_input_names = None

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
    ):

        if framework is None:
            framework = get_framework(model)

        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.device = device if framework == "tf" else torch.device("cpu" if device < 0 else "cuda:{}".format(device))
        self.binary_output = binary_output

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
        if self.framework == "tf":
            import tensorflow as tf
            with tf.device("/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)):
                yield
        else:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (:obj:`List[str]` or :obj:`dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models = [item[1].__name__ for item in supported_models.items()]
        if self.model.__class__.__name__ not in supported_models:
            raise PipelineException(
                self.task,
                self.model.base_model_prefix,
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}",
            )

    def _parse_and_tokenize(self, inputs, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching

        Args:
            inputs: dict holding all the keyword arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array

        Returns:
            Numpy array
        """
        # Encode for forward
        with self.device_placement():
            if self.framework == "tf":
                # TODO trace model
                predictions = self.model(inputs.data, training=False)[0]
            else:
                with torch.no_grad():
                    inputs = self.ensure_tensor_on_device(**inputs)
                    predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()


# Can't use @add_end_docstrings(PIPELINE_INIT_ARGS) here because this one does not accept `binary_output`
class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    This feature extraction pipeline can currently be loaded from :func:`~transformers.pipeline` using the task
    identifier: :obj:`"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    `huggingface.co/models <https://huggingface.co/models>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id.
    """

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            args_parser: ArgumentHandler = None,
            device: int = -1,
            task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (:obj:`str` or :obj:`List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of :obj:`float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs).tolist()


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return all prediction scores or just the one of the predicted class.
    """,
)
class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any :obj:`ModelForSequenceClassification`. See the `sequence classification
    examples <../task_summary.html#sequence-classification>`__ for more information.

    This text classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"sentiment-analysis"` (for classifying sequences according to positive or negative
    sentiments).

    If multiple classification labels are available (:obj:`model.config.num_labels >= 2`), the pipeline will run a
    softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=text-classification>`__.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        )

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.

            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        outputs = super().__call__(*args, **kwargs)

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using a :obj:`ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for `entailment` is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the `entailment` label must be included in the model
    config's :attr:`~transformers.PretrainedConfig.label2id`.

    This NLI pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task identifier:
    :obj:`"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on `huggingface.co/models <https://huggingface.co/models?search=nli>`__.
    """

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args_parser = args_parser
        if self.entailment_id == -1:
            logger.warning(
                "Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )

    @property
    def entailment_id(self):
        for label, ind in self.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1

    def _parse_and_tokenize(
            self, sequences, candidate_labels, hypothesis_template, padding=True, add_special_tokens=True, **kwargs
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        sequence_pairs = self._args_parser(sequences, candidate_labels, hypothesis_template)
        inputs = self.tokenizer(
            sequence_pairs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation="only_first",
        )

        return inputs

    def __call__(
            self,
            sequences: Union[str, List[str]],
            candidate_labels,
            hypothesis_template="This example is {}.",
            multi_class=False,
    ):
        """
        Classify the sequence(s) given as inputs. See the :obj:`~transformers.ZeroShotClassificationPipeline`
        documentation for more information.

        Args:
            sequences (:obj:`str` or :obj:`List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (:obj:`str` or :obj:`List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (:obj:`str`, `optional`, defaults to :obj:`"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is :obj:`"This example is {}."` With the candidate label :obj:`"sports"`, this would be fed
                into the model like :obj:`"<cls> sequence to classify <sep> This example is sports . <sep>"`. The
                default template works well in many cases, but it may be worthwhile to experiment with different
                templates depending on the task setting.
            multi_class (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not multiple candidate labels can be true. If :obj:`False`, the scores are normalized such
                that the sum of the label likelihoods for each sequence is 1. If :obj:`True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        if sequences and isinstance(sequences, str):
            sequences = [sequences]

        outputs = super().__call__(sequences, candidate_labels, hypothesis_template)
        num_sequences = len(sequences)
        candidate_labels = self._args_parser._parse_labels(candidate_labels)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        if len(candidate_labels) == 1:
            multi_class = True

        if not multi_class:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]

        result = []
        for iseq in range(num_sequences):
            top_inds = list(reversed(scores[iseq].argsort()))
            result.append(
                {
                    "sequence": sequences if isinstance(sequences, str) else sequences[iseq],
                    "labels": [candidate_labels[i] for i in top_inds],
                    "scores": scores[iseq][top_inds].tolist(),
                }
            )

        if len(result) == 1:
            return result[0]
        return result


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        top_k (:obj:`int`, defaults to 5): The number of predictions to return.
    """,
)
class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any :obj:`ModelWithLMHead`. See the `masked language modeling
    examples <../task_summary.html#masked-language-modeling>`__ for more information.

    This mask filling pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=masked-lm>`__.

    .. note::

        This pipeline only works for inputs with exactly one token masked.
    """

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            args_parser: ArgumentHandler = None,
            device: int = -1,
            top_k=5,
            task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

        self.check_model_type(MODEL_FOR_MASKED_LM_MAPPING)
        self.top_k = top_k

    def ensure_exactly_one_mask_token(self, masked_index: np.ndarray):
        numel = np.prod(masked_index.shape)
        if numel > 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"More than one mask_token ({self.tokenizer.mask_token}) is not supported",
            )
        elif numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def __call__(self, *args, targets=None, top_k: Optional[int] = None, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (:obj:`str` or :obj:`List[str]`, `optional`):
                When passed, the model will return the scores for the passed token or tokens rather than the top k
                predictions in the entire vocabulary. If the provided targets are not in the model vocab, they will be
                tokenized and the first resulting token will be used (with a warning).
            top_k (:obj:`int`, `optional`):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (:obj:`str`) -- The corresponding input with the mask token prediction.
            - **score** (:obj:`float`) -- The corresponding probability.
            - **token** (:obj:`int`) -- The predicted token id (to replace the masked one).
            - **token** (:obj:`str`) -- The predicted token (to replace the masked one).
        """
        inputs = self._parse_and_tokenize(*args, **kwargs)
        outputs = self._forward(inputs, return_tensors=True)

        results = []
        batch_size = outputs.shape[0] if self.framework == "tf" else outputs.size(0)

        if targets is not None:
            if len(targets) == 0 or len(targets[0]) == 0:
                raise ValueError("At least one target must be provided when passed.")
            if isinstance(targets, str):
                targets = [targets]

            targets_proc = []
            for target in targets:
                target_enc = self.tokenizer.tokenize(target)
                if len(target_enc) > 1 or target_enc[0] == self.tokenizer.unk_token:
                    logger.warning(
                        "The specified target token `{}` does not exist in the model vocabulary. Replacing with `{}`.".format(
                            target, target_enc[0]
                        )
                    )
                targets_proc.append(target_enc[0])
            target_inds = np.array(self.tokenizer.convert_tokens_to_ids(targets_proc))

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            if self.framework == "tf":
                logger.warning('not impl with tf')
            else:
                masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)

                # Fill mask pipeline supports only one ${mask_token} per sample
                self.ensure_exactly_one_mask_token(masked_index.numpy())

                logits = outputs[i, masked_index.item(), :]
                probs = logits.softmax(dim=0)
                if targets is None:
                    values, predictions = probs.topk(top_k if top_k is not None else self.top_k)
                else:
                    values = probs[..., target_inds]
                    sort_inds = list(reversed(values.argsort(dim=-1)))
                    values = values[..., sort_inds]
                    predictions = target_inds[sort_inds]

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens),
                        "score": v,
                        "token": p,
                        "token_str": self.tokenizer.convert_ids_to_tokens(p),
                    }
                )

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results


class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, *args, **kwargs):

        if args is not None and len(args) > 0:
            inputs = list(args)
            batch_size = len(inputs)
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError("offset_mapping should have the same batch size as the input")
        return inputs, offset_mapping


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        ignore_labels (:obj:`List[str]`, defaults to :obj:`["O"]`):
            A list of labels to ignore.
        grouped_entities (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group the tokens corresponding to the same entity together in the predictions or not.
    """,
)
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            args_parser: ArgumentHandler = TokenClassificationArgumentHandler(),
            device: int = -1,
            binary_output: bool = False,
            ignore_labels=["O"],
            task: str = "",
            grouped_entities: bool = False,
            ignore_subwords: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self.check_model_type(
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities
        self.ignore_subwords = ignore_subwords

        if self.ignore_subwords and not self.tokenizer.is_fast:
            raise ValueError(
                "Slow tokenizers cannot ignore subwords. Please set the `ignore_subwords` option"
                "to `False` or use a fast tokenizer."
            )

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `grouped_entities` is set to True.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []

        for i, sentence in enumerate(inputs):

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                )
                if self.tokenizer.is_fast:
                    offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens.data)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            # Filter to labels not in `self.ignore_labels`
            # Filter special_tokens
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if (self.model.config.id2label[label_idx] not in self.ignore_labels) and not special_tokens_mask[idx]
            ]

            for idx, label_idx in filtered_labels_idx:
                if offset_mapping is not None:
                    start_ind, end_ind = offset_mapping[idx]
                    word_ref = sentence[start_ind:end_ind]
                    word = self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]
                    is_subword = len(word_ref) != len(word)

                    if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                        word = word_ref
                        is_subword = False
                else:
                    word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

                    start_ind = None
                    end_ind = None

                entity = {
                    "word": word,
                    "score": score[idx][label_idx].item(),
                    "entity": self.model.config.id2label[label_idx],
                    "index": idx,
                    "start": start_ind,
                    "end": end_ind,
                }

                if self.grouped_entities and self.ignore_subwords:
                    entity["is_subword"] = is_subword

                entities += [entity]

            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                    (
                            entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                            and entity["entity"].split("-")[0] != "B"
                    )
                    and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups


NerPipeline = TokenClassificationPipeline

# Register all the supported tasks here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "pt": AutoModel if is_torch_available() else None,
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "ner": {
        "impl": TokenClassificationPipeline,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "pt": AutoModelForMaskedLM if is_torch_available() else None,
        "default": {"model": {"pt": "distilroberta-base", "tf": "distilroberta-base"}},
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "config": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "tokenizer": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
        },
    },
}


def check_task(task: str) -> Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`
            - :obj:`"sentiment-analysis"`
            - :obj:`"ner"`
            - :obj:`"question-answering"`
            - :obj:`"fill-mask"`
            - :obj:`"summarization"`
            - :obj:`"translation_xx_to_yy"`
            - :obj:`"translation"`
            - :obj:`"text-generation"`
            - :obj:`"conversational"`

    Returns:
        (task_defaults:obj:`dict`, task_options: (:obj:`tuple`, None)) The actual dictionary required to initialize the
        pipeline and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None

    if task.startswith("translation"):
        tokens = task.split("_")
        if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
            targeted_task = SUPPORTED_TASKS["translation"]
            return targeted_task, (tokens[1], tokens[3])
        raise KeyError("Invalid translation task {}, use 'translation_XX_to_YY' format".format(task))

    raise KeyError(
        "Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys()) + ["translation_XX_to_YY"])
    )


def pipeline(
        task: str,
        model: Optional = None,
        config: Optional[Union[str, PretrainedConfig]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        framework: Optional[str] = None,
        revision: Optional[str] = None,
        use_fast: bool = True,
        **kwargs
) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"sentiment-analysis"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"ner"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"fill-mask"`: will return a :class:`~transformers.FillMaskPipeline`.
            - :obj:`"summarization"`: will return a :class:`~transformers.SummarizationPipeline`.
            - :obj:`"translation_xx_to_yy"`: will return a :class:`~transformers.TranslationPipeline`.
            - :obj:`"text2text-generation"`: will return a :class:`~transformers.Text2TextGenerationPipeline`.
            - :obj:`"text-generation"`: will return a :class:`~transformers.TextGenerationPipeline`.
            - :obj:`"zero-shot-classification:`: will return a :class:`~transformers.ZeroShotClassificationPipeline`.
            - :obj:`"conversation"`: will return a :class:`~transformers.ConversationalPipeline`.
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from :class:`~transformers.PreTrainedModel` (for PyTorch)
            or :class:`~transformers.TFPreTrainedModel` (for TensorFlow).

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default configuration file for the requested model will be used. That means that if
            :obj:`model` is given, its default configuration will be used. However, if :obj:`model` is not supplied,
            this :obj:`task`'s default model's config is used instead.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default tokenizer for the given :obj:`model` will be loaded (if it is a string). If
            :obj:`model` is not specified or not a string, then the default tokenizer for :obj:`config` is loaded (if
            it is a string). However, if :obj:`config` is also not given or not a string, then the default tokenizer
            for the given :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
        use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use a Fast tokenizer if possible (a :class:`~transformers.PreTrainedTokenizerFast`).
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> pipeline('sentiment-analysis')

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> pipeline('ner', model=model, tokenizer=tokenizer)
    """
    # Retrieve the task
    targeted_task, task_options = check_task(task)

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model = get_default_model(targeted_task, framework, task_options)

    framework = framework or get_framework(model)

    task_class, model_class = targeted_task["impl"], targeted_task[framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            use_fast = tokenizer[1].pop("use_fast", use_fast)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer[0], use_fast=use_fast, revision=revision, **tokenizer[1]
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision, use_fast=use_fast)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard, revision=revision)

    # Instantiate model if needed
    if isinstance(model, str):
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        elif framework == "tf" and model.endswith(".bin"):
            model_kwargs["from_pt"] = True
            logger.warning(
                "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                "Trying to load the model with Tensorflow."
            )

        if model_class is None:
            raise ValueError(
                f"Pipeline using {framework} framework, but this framework is not supported by this pipeline."
            )

        model = model_class.from_pretrained(model, config=config, revision=revision, **model_kwargs)
        if task == "translation" and model.config.task_specific_params:
            for key in model.config.task_specific_params:
                if key.startswith("translation"):
                    task = key
                    warnings.warn(
                        '"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{}"'.format(
                            task
                        ),
                        UserWarning,
                    )
                    break

    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard, framework=framework, task=task, **kwargs)

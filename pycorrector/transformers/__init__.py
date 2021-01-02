# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

__version__ = "4.2.0dev0"

from pycorrector.utils.logger import logger

# Configuration
from .configuration_utils import PretrainedConfig

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)

# Model Cards
from .modelcard import ModelCard
from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .models.auto import (
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CONFIG_MAPPING,
    MODEL_NAMES_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoTokenizer,
)
from .models.bart import BartConfig, BartTokenizer
from .models.bert import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BasicTokenizer,
    BertConfig,
    BertTokenizer,
    WordpieceTokenizer,
)
from .models.bert_generation import BertGenerationConfig
from .models.distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig, DistilBertTokenizer
from .models.electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraTokenizer
from .models.encoder_decoder import EncoderDecoderConfig
from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer

# Pipelines
from .pipelines import (
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)

# Tokenization
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)



if is_sentencepiece_available():
    from .models.albert import AlbertTokenizer
    from .models.bert_generation import BertGenerationTokenizer

if is_tokenizers_available():
    from .models.albert import AlbertTokenizerFast
    from .models.bart import BartTokenizerFast
    from .models.bert import BertTokenizerFast
    from .models.distilbert import DistilBertTokenizerFast
    from .models.electra import ElectraTokenizerFast
    from .models.gpt2 import GPT2TokenizerFast
    from .models.roberta import RobertaTokenizerFast
    from .tokenization_utils_fast import PreTrainedTokenizerFast

    if is_sentencepiece_available():
        from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer


# Modeling
from .generation_beam_search import BeamScorer, BeamSearchScorer
from .generation_logits_process import (
    HammingDiversityLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    LogitsWarper,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .generation_utils import top_k_top_p_filtering
from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
from .models.albert import (
    ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
    AlbertPreTrainedModel,
    load_tf_weights_in_albert,
)
from .models.auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
)
from .models.bart import (
    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
    BartPretrainedModel,
    PretrainedBartModel,
)
from .models.bert import (
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLayer,
    BertLMHeadModel,
    BertModel,
    BertPreTrainedModel,
    load_tf_weights_in_bert,
)
from .models.bert_generation import (
    BertGenerationDecoder,
    BertGenerationEncoder,
    load_tf_weights_in_bert_generation,
)
from .models.distilbert import (
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from .models.electra import (
    ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
    ElectraPreTrainedModel,
    load_tf_weights_in_electra,
)
from .models.encoder_decoder import EncoderDecoderModel
from .models.roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .models.xlm import (
    XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    XLMForMultipleChoice,
    XLMForQuestionAnswering,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMPreTrainedModel,
    XLMWithLMHeadModel,
)


if is_flax_available():
    from .modeling_flax_utils import FlaxPreTrainedModel
    from .models.auto import FLAX_MODEL_MAPPING, FlaxAutoModel
    from .models.bert import FlaxBertForMaskedLM, FlaxBertModel
    from .models.roberta import FlaxRobertaModel


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )

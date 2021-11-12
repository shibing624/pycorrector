"""
@Time   :   2021-01-21 10:37:36
@File   :   defaults.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cpu"
_C.MODEL.GPU_IDS = [0]
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.BERT_CKPT = 'bert-base-chinese'
_C.MODEL.NAME = ''
_C.MODEL.WEIGHTS = ''
_C.MODEL.HYPER_PARAMS = []

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Max length of input text.
_C.INPUT.MAX_LEN = 512


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
# List of the dataset names for validation, as present in paths_catalog.py
_C.DATASETS.VALID = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "AdamW"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.9999
_C.SOLVER.STEPS = (10,)
_C.SOLVER.SCHED = "WarmupExponentialLR"
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 2
_C.SOLVER.WARMUP_EPOCHS = 1024
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.DELAY_ITERS = 0
_C.SOLVER.ETA_MIN_LR = 3e-7
_C.SOLVER.MAX_ITER = 10
_C.SOLVER.INTERVAL = 'step'

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.ACCUMULATE_GRAD_BATCHES = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.BATCH_SIZE = 16


_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8
_C.TEST.CKPT_FN = ""

# ---------------------------------------------------------------------------- #
# Task specific
# ---------------------------------------------------------------------------- #
_C.TASK = CN()
_C.TASK.NAME = "CSC"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.MODE = ['train', 'test']



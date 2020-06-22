#!/bin/bash
# Author: XuMing(xuming624@qq.com)
# Brief: Train conv seq2seq model

GPU_ID=0
DATA_BIN_DIR="output/bin/"
OUT_DIR="output/models"
BATCH_SIZE=64
MAX_LEN=400
MAX_TOKENS=4096
SEED=1

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m fairseq_cli.train \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    -a fconv \
    --num-workers=4 --skip-invalid-size-inputs-valid-test \
    --encoder-embed-dim 500 \
    --decoder-embed-dim 500 \
    --decoder-out-embed-dim 500 \
    --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' \
    --dropout='0.2' --clip-norm=0.1 \
    --optimizer nag --momentum 0.99 \
    --lr-scheduler=reduce_lr_on_plateau --lr=0.25 --lr-shrink=0.1 --min-lr=1e-4 \
    --max-epoch 100 \
    --max-sentences ${BATCH_SIZE} \
    --max-tokens ${MAX_TOKENS} \
    --seed ${SEED}

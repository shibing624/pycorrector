#!/bin/bash
# Author: XuMing(xuming624@qq.com)
# Brief: Train transformer model

GPU_ID=0
DATA_BIN_DIR="output/bin/"
OUT_DIR="output/models/"
BATCH_SIZE=64
MAX_LEN=400
MAX_TOKENS=4096
SEED=1
# MAX_UPDATE=300000 #default
MAX_UPDATE=10

CUDA_VISIBLE_DEVICES="${GPU_ID}" fairseq-train \
    ${DATA_BIN_DIR} \
    --save-dir ${OUT_DIR} \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed ${SEED} \
    --max-tokens ${MAX_TOKENS} \
    --save-interval-updates 10000 \
    --max-update ${MAX_UPDATE}

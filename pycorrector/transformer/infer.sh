#!/bin/bash
# Author: XuMing(xuming624@qq.com)
# Brief: infer

DATA_BIN_DIR="output/bin/"
MODEL="output/models/checkpoint_best.pt"
BEAM=1
BATCH_SIZE=64

fairseq-generate \
    ${DATA_BIN_DIR} \
    --gen-subset valid \
    --task translation_lev \
    --path ${MODEL} \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam ${BEAM} --remove-bpe \
    --print-step \
    --batch-size ${BATCH_SIZE}

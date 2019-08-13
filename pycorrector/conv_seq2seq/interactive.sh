#!/bin/bash
# Author: XuMing <xuming624@qq.com>
# Brief: infer with conv seq2seq model

GPU_ID=0
INPUT_FILE="output/valid.src"
OUTPUT_FILE="output/beamserch_out.txt"
DATA_BIN_DIR="output/bin/"
MODEL="output/models/checkpoint_best.pt"
BEAM=5
NBEST=${BEAM}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m fairseq_cli.interactive \
    --path ${MODEL} \
    --beam ${BEAM} --nbest ${NBEST} \
    --model-overrides "{'encoder_embed_path': None, 'decoder_embed_path': None}" \
    ${DATA_BIN_DIR} < ${INPUT_FILE} > ${OUTPUT_FILE}
# HighLight Prompt: False / CoOp: False (Zero-shot)

#!/bin/bash

# custom config
DATA=/workspace
TRAINER=EVoCLIP

GPU=$1
DATASET=$2
CFG=$3  # config file
FRAME=$4

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${DATASET}/${CFG}.yaml \
    --output-dir output/${DATASET}/zeroshot/${CFG}/${FRAME}frames_val \
    --eval-only \
    MODEL.EVO.ENABLE False \
    INPUT.FRAMES ${FRAME}
done
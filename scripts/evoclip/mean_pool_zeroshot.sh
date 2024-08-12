#!/bin/bash

# custom config
DATA=/workspace
TRAINER=MeanPoolZeroshotCLIP

GPU=$1
DATASET=$2
CFG=$3  # config file

for SEED in 1
do
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/HIGHLIGHT/${DATASET}/${CFG}.yaml \
    --output-dir output/${DATASET}/zeroshot/${TRAINER}/${CFG} \
    --eval-only \
    MODEL.HIGHLIGHT.ENABLE False \
    MODEL.COOP.ENABLE False \
    DATASET.NUM_SHOTS 8
done
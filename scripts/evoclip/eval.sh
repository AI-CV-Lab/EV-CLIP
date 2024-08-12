#!/bin/bash

#cd ../..

# custom config
DATA=/workspace
TRAINER=EVoCLIP

GPU=$1
DATASET=$2
CFG=$3      # config file
SHOTS=$4    # number of shots (1, 2, 4, 8, 16) / full (-1)
EPOCH=$5    # load-epoch (null is best)

for SEED in 1 2 3
do
    if [ ${EPOCH} ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}/${CFG}.yaml \
        --output-dir output/evaluation/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
        --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
        --load-epoch ${EPOCH} \
        --eval-only \
        SHOT_DIR "/home/Datasets" \
        DATASET.NUM_SHOTS ${SHOTS}
    else
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}/${CFG}.yaml \
        --output-dir output/evaluation/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
        --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
        --eval-only \
        SHOT_DIR "/home/Datasets" \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

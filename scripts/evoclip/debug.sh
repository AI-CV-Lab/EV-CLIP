#!/bin/bash

#cd ../..

# custom config
DATA=/workspace
TRAINER=EVoCLIP

GPU=$1
DATASET=$2
CFG=$3      # config file
FRAME=$4    # number of frames (8, 16, 32, 64)
SHOTS=$5    # number of shots (1, 2, 4, 8, 16) / full (-1)
BATCH=$6
ENC_OUT_DIM=$7
DECODER=$8
PROMPT_AGGREGATION=$9
PROMPT_INIT=$10
TEMPORAL_AGGREGATION=$11

for SEED in 1
do
    DIR=output/${TRAINER}/${DATASET}/${CFG}_${FRAME}frames_${SHOTS}shots/batch_${BATCH}/EVO_True_CoOp_False/encoder_${ENC_OUT_DIM}/dec${DECODER}/prompt-${PROMPT_AGGREGATION}_init-${PROMPT_INIT}_${TEMPORAL_AGGREGATION}/seed${SEED}
    if [ -d "$DIR" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}/${CFG}.yaml \
        --model-dir ${DIR}
        SHOT_DIR "/home/Datasets" \
    else
        echo "Oops! The results don't exist at ${DIR}"
    fi
done
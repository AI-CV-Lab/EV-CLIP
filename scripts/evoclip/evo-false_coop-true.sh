# HighLight Prompt: False / CoOp: True

#!/bin/bash

#cd ../..

# custom config
DATA=/home/Datasets
TRAINER=EVoCLIP

GPU=$1
DATASET=$2
CFG=$3      # config file
FRAME=$4    # number of frames (8, 16, 32, 64)
SHOTS=$5    # number of shots (1, 2, 4, 8, 16) / full (-1)
TEMPORAL_AGGREGATION=$6

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${FRAME}frames_${SHOTS}shots/EVO_False_CoOp_True/encoder_${ENC_OUT_DIM}/${TEMPORAL_AGGREGATION}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}/${CFG}.yaml \
        --output-dir ${DIR} \
        SHOT_DIR "/home/Datasets" \
        INPUT.FRAMES ${FRAME} \
        DATASET.NUM_SHOTS ${SHOTS} \
        MODEL.EVO.ENABLE False \
        MODEL.COOP.ENABLE True \
        MODEL.TEMPORAL ${TEMPORAL_AGGREGATION}
    fi
done
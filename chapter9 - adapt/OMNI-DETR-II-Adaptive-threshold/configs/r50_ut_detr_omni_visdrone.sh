#!/usr/bin/env bash

set -x

EXP_DIR=results/visDrone2021_det/11-20-classes/900-queries-16-att-heads-4-feature-levels-20-burn-in-50-50-split/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label '50-50-label_scaled_annotations_visdrone_train.json_w_indicator.json' \
    --annotation_json_unlabel '50-50-unlabel_scaled_annotations_visdrone_train_w_indicator.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '../../dataset/visDrone2022_det_v3' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'visDrone' \
    # --resume 'results/visDrone2021_det/11-20-classes/900-queries-16-att-heads-4-feature-levels-50-burn-in-50-50-split/checkpoint0055.pth' 


    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # configs/r50_ut_detr_omni_visdrone.sh --resume results/visDrone2021_det/11-20-classes/900-queries-16-att-heads-4-feature-levels-50-burn-in-50-50-split/checkpoint0055.pth --eval
#!/usr/bin/env bash

set -x

EXP_DIR=results/sdugamod/ground/11-20-classes/20-burn-in-50-50-split-DEBUG/
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'ground_train_aligned_ids_w_indicator.json' \
    --annotation_json_unlabel 'Ground_8605_scaled_labels_ids_w_indicator.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '../../SDU-GAMODv3' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'sdugamod_ground' \
    --resume 'results/sdugamod/ground/11-20-classes/20-burn-in-50-50-split/checkpoint0027.pth' 

    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # configs/r50_ut_detr_omni_visdrone.sh --resume results/visDrone2021_det/11-20-classes/900-queries-16-att-heads-4-feature-levels-50-burn-in-50-50-split/checkpoint0055.pth --eval
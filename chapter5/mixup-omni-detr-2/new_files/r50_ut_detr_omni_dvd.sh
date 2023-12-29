#!/usr/bin/env bash

set -x

EXP_DIR=results/sdugamod/ground/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'aerial_train_aligned_ids_w_indicator.json' \
    --annotation_json_unlabel 'Ground_8605_scaled_labels_ids_w_indicator.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '/data/jlorray1/SDU-GAMODv4' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'sdugamod_aerial' \

# REALLY not sure about the --annotation_json_unlabel setting
# If anything breaks path wise, it's probably the data_path setting
# REALLY not sure about the dataset_file setting
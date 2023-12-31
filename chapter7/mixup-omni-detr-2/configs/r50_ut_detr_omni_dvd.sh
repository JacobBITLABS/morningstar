#!/usr/bin/env bash

set -x

EXP_DIR=results/sdugamod/aerial/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_aerial_label 'aerial_train_aligned_ids_w_indicator_with_perspective_with_points.json' \
    --annotation_json_aerial_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --annotation_json_ground_label 'ground_train_aligned_ids_w_indicator_with_perspective_with_points.json' \
    --annotation_json_ground_unlabel 'Ground_8605_scaled_labels_ids_w_indicator_aligned_ids.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path '../../DATASET/SDU-GAMODv4-old/' \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --pixels 600 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'dvd' \
#    --annotation_json_label 'aerial/aligned_ids/aerial_train_aligned_ids_w_indicator.json' \
#    --annotation_json_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
#    --annotation_json_aerial_label 'aerial/aligned_ids/aerial_train_aligned_ids_w_indicator.json' \
#    --annotation_json_aerial_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
#    --annotation_json_ground_label 'ground/aligned_ids/ground_train_aligned_ids_w_indicator.json' \
#    --annotation_json_ground_unlabel 'ground_8605_scaled_labels_w_indicator_aligned_ids.json' \
#    --dataset_file '/data/jlorray1/SDU-GAMODv4' \
#    --dataset_file 'sdugamod_aerial' \
#    --data_path './dvd' \

# REALLY not sure about the --annotation_json_unlabel setting
# REALLY not sure about the dataset_file setting
# If anything breaks path wise, it's probably the data_path setting

#!/usr/bin/env bash

set -x

EXP_DIR=results/vanilla_omni_detr/
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --data_path '../../SDU-GAMODv4-old/' \
    --label_label_root 'supervised_annotations/aerial/aligned_ids/'\
    --unlabel_label_root '../../SDU-GAMODv3/unsupervised_annotations/'\
    --annotation_json_train_label 'aerial_train_aligned_ids_w_indicator_with_perspective_with_points.json' \
    --annotation_json_train_unlabel 'Aerial_8605_scaled_labels_w_indicator_aligned_ids.json' \
    --annotation_json_val 'aerial_valid_aligned_ids_w_indicator.json' \
    --data_dir_label_train 'scaled_dataset/train/droneview' \
    --data_dir_label_val 'scaled_dataset/val/droneview' \
    --data_dir_unlabel_train '../../SDU-GAMODv3/unsupervised_annotations/aerial/' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --lr 2e-4 \
    --epochs 58 \
    --lr_drop 150 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --save_freq 4 \
    --dataset_file 'dvd' \
    --sam \
    --resume '../omni-detr/results/sdugamod/11-20-classes/20-burn-in-50-50-split/checkpoint0019.pth' #'results/tests_2/checkpoint0019.pth'


    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/r50_ut_detr_omni_dvd.sh
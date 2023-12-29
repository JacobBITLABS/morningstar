#!/usr/bin/env bash

set -x

EXP_DIR=results/gamod/ground-aerial/
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file 'gamod' \
    --data_path '../../SDU-GAMODv3/' \
    --epochs 58 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    ${PY_ARGS}
    # --resume 'results/visDrone_scaled_600/visDrone2021_mot/11-20-classes/900-queries-16-att-heads-4-feature-levels-test_val_swap/checkpoint0049.pth' \
    
    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/
    
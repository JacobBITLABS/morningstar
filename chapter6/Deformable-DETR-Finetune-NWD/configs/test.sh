#!/usr/bin/env bash

set -x

EXP_DIR=exps/Mixup-random-beta-alpha-1-1-lam
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH
#python -u -m trace --trace main.py \

# If issues with Dataloaders, set the number of workers to 0 and use these two before python... 
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1


python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file 'gamod' \
    --data_path '../../SDU-GAMODv3/' \
    --epochs 39 \
    --num_queries 900 \
    --nheads 16 \
    --num_feature_levels 4 \
    --resume 'exps/Mixup-swap-image-blend-order/checkpoint0024.pth' \
    --eval 
    ${PY_ARGS}
    
    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/

    # eval 
    # <path to config file> --resume <path to pre-trained model> --eval
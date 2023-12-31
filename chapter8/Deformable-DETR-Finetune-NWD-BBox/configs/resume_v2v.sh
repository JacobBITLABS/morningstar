# #!/usr/bin/env bash

# set -x

# EXP_DIR=results/v2v/
# PY_ARGS=${@:1}

# export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

# python3 -u main.py \
#     --output_dir ${EXP_DIR} \
#     --dataset_file 'dvd' \
#     --data_path '../../DATASET/SDU-GAMODv4-old/' \
#     --num_queries 900 \
#     --nheads 16 \
#     --epochs 49 \
#     --num_feature_levels 4 \
#     # --eval \
#     ${PY_ARGS}
    
#     # --data_path '../../DATASET/SDU-GAMODv4-old/' \
#     #--finetune True \
#     # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
#     # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/gamod_r50_deformable_detr.sh
#     # --resume 'results/gamod/6kG8kA/checkpoint0014.pth' \


#!/usr/bin/env bash

set -x

EXP_DIR=results/v2v-v5/
PY_ARGS=${@:1}

export LD_LIBRARY_PATH=/work/mconda3/envs/deformable_detr/lib:$LD_LIBRARY_PATH

# python3 -u main.py \
#     --output_dir ${EXP_DIR} \
#     --dataset_file 'dvd' \
#     --data_path '../../SDU-GAMODv4-old/' \
#     --num_queries 900 \
#     --nheads 16 \
#     --epochs 49 \
#     --num_feature_levels 4 \
#     --resume_ground '../Deformable-DETR/results/gamod/ground/checkpoint0039.pth' \
#     --resume_aerial '../Deformable-DETR/results/gamod/aerial/checkpoint0039.pth' \
#     # --eval \
#     ${PY_ARGS}
    
    # --data_path '../../DATASET/SDU-GAMODv4-old/' \
    #--finetune True \
    # GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/
    # GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/gamod_r50_deformable_detr.sh
    # --resume 'results/gamod/6kG8kA/checkpoint0014.pth' \


python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file 'dvd' \
    --data_path '../../DATASET/SDU-GAMODv4-old/' \
    --num_queries 900 \
    --nheads 16 \
    --epochs 49 \
    --num_feature_levels 4 \
    --resume_ground '../Deformable-DETR-Finetune/results/gamod/ground/checkpoint0039.pth' \
    --resume_aerial '../Deformable-DETR-Finetune/results/gamod/aerial/checkpoint0039.pth' \
    # --eval \
    ${PY_ARGS}



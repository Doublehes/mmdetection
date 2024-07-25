#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

project_dir=/home/shuang.he/workspace/mmdetection

cd $project_dir

PYTHONPATH=$project_dir:$PYTHONPATH \
/home/shuang.he/miniconda3/envs/openmmlab/bin/python \
    -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $project_dir/tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
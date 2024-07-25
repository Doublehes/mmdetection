#!/usr/bin/env bash

CONFIG=$1

project_dir=/home/shuang.he/workspace/mmdetection
py_bin=/home/shuang.he/miniconda3/envs/openmmlab/bin/python

cd $project_dir

PYTHONPATH=$project_dir:$PYTHONPATH $py_bin $project_dir/tools/train.py $CONFIG
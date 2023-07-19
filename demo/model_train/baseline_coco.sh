#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
CONFIG=configs/fcos_semi/r50_caffe_mslonger_tricks_0.Xdata.py
WORKDIR=workdir_coco/r50_caffe_mslonger_tricks_0.1data
GPU=1

CUDA_VISIBLE_DEVICES=0 PORT=29507 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR

#!/bin/bash
# One RoboTwin 2.0 task, all instruction arms, one model load (run_instruction_arms.py).
# usage: run_task.sh TASK EPISODES SEED VIDEO_EPISODES ARMS(comma) RUN_TAG
TASK=$1; EPS=$2; SEED=$3; VEPS=$4; ARMS=$5; TAG=$6
D=/home/jingwu/workspace/FlashRT/research/topics/long-horizon-planner-executor/experiments/robotwin_instruction
cd /home/jingwu/workspace/FlashRT && source tmp/env_5090.sh && source tmp/robotwin_env.sh
export PATH=$D/bin:$PATH   # bin/ffmpeg -> imageio_ffmpeg's static ffmpeg (eval videos)
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd $ROBOTWIN_ROOT
$ROBOTWIN_PY -u $D/run_instruction_arms.py $TASK $EPS $SEED $D/runs/$TAG $D/videos/$TAG $VEPS $ARMS

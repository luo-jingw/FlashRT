#!/bin/bash
# R1 remainder after the long-context planner ran out of memory next to an executor holding its text encoder:
# every executor here frees the Qwen3-4B text encoder after set_prompt (DROP_TEXT_ENCODER=1).
# Reduced MPS grid: planner rate {1 Hz, continuous} at short context, continuous at long context.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 DROP_TEXT_ENCODER=1 CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
D=60
run() { python colocation_bench.py "$@" 2>&1 | grep -E "hz=|Error" ; }
mkdir -p r1 $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export PLANNER_CTX_EXTRA=0
run prio 0 $D r1/ctx0_prio_hz0_droptext.json
export PLANNER_CTX_EXTRA=16000
for hz in 1 2 0; do run serial $hz $D r1/ctx16000_serial_hz${hz}.json; done
for hz in 0.5 1 2 0; do for m in process prio; do run $m $hz $D r1/ctx16000_${m}_hz${hz}.json; done; done
unset PLANNER_CTX_EXTRA
nvidia-cuda-mps-control -d
for p in 25 50; do CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100-p)) run isolated 0 $D r1/mps_exec$((100-p))_alone.json; done
for cell in "0 1" "0 0" "16000 0"; do
  set -- $cell; export PLANNER_CTX_EXTRA=$1
  for p in 25 50 100; do
    if [ $p = 100 ]; then e=100; else e=$((100-p)); fi
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$e run mps$p $2 $D r1/ctx$1_mps${p}_hz$2.json
  done
done
echo quit | nvidia-cuda-mps-control
sleep 3
unset PLANNER_CTX_EXTRA
run isolated 0 $D r1/isolated_end.json
echo R1_DONE

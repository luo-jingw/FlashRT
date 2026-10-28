#!/bin/bash
# R1 grid: executor period 50 ms; planner modes x planner rate; short (~260 token) and long (~16 k token) planner context.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
D=60
run() { python colocation_bench.py "$@" 2>&1 | grep "hz=" ; }
mkdir -p r1 $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
run isolated 0 $D r1/isolated_start.json
for ctx in 0 16000; do
  export PLANNER_CTX_EXTRA=$ctx
  for hz in 0.5 1 2 0; do
    for m in serial process prio; do run $m $hz $D r1/ctx${ctx}_${m}_hz${hz}.json; done
  done
done
unset PLANNER_CTX_EXTRA
nvidia-cuda-mps-control -d
for p in 25 50; do CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100-p)) run isolated 0 $D r1/mps_exec$((100-p))_alone.json; done
for ctx in 0 16000; do
  export PLANNER_CTX_EXTRA=$ctx
  for hz in 0.5 1 2 0; do
    for p in 25 50 100; do
      if [ $p = 100 ]; then e=100; else e=$((100-p)); fi
      CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$e run mps$p $hz $D r1/ctx${ctx}_mps${p}_hz${hz}.json
    done
  done
done
echo quit | nvidia-cuda-mps-control
sleep 3
unset PLANNER_CTX_EXTRA
run isolated 0 $D r1/isolated_end.json
echo R1_DONE

#!/bin/bash
# R4: finish the MPS SM-limit trade-off: planner limited to 10 % and 15 % of SMs (executor 90 % / 85 %), continuous
# planner, short and long context; executor alone at the same SM limits as the bandwidth-interference baseline.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 DROP_TEXT_ENCODER=1 CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
D=60
run() { python colocation_bench.py "$@" 2>&1 | grep -E "hz=|Error" ; }
mkdir -p r1 $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
for p in 10 15; do CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100-p)) run isolated 0 $D r1/mps_exec$((100-p))_alone.json; done
for ctx in 0 16000; do
  export PLANNER_CTX_EXTRA=$ctx
  for p in 10 15; do CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100-p)) run mps$p 0 $D r1/ctx${ctx}_mps${p}_hz0.json; done
done
echo quit | nvidia-cuda-mps-control
echo R4_DONE

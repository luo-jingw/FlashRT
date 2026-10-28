#!/bin/bash
# R3: (a) session state and switching on one frontend; (b) N executor processes, each a session at 20 Hz, sharing the GPU
# by time-slicing and under MPS (no SM limit). N <= 2: one executor without its text encoder reserves 15.3 GB of the 32 GB card.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
mkdir -p r3
[ -z "$R3_SKIP_A" ] && python session_switch.py r3/session_switch.json 2>&1 | grep "^{"
nproc_run() {  # $1 = tag, $2 = N
  local at=$(( $(date +%s) + 60 * $2 + 90 )) pids=()
  for i in $(seq 1 $2); do
    DROP_TEXT_ENCODER=1 START_AT=$at python colocation_bench.py isolated 0 60 r3/$1_n$2_p$i.json > r3/$1_n$2_p$i.log 2>&1 & pids+=($!)
    sleep 60  # stagger loads: each process peaks at ~23 GB (text encoder resident) before freeing it
  done
  wait "${pids[@]}"
  grep -h "hz=" r3/$1_n$2_p*.log | sed "s/^/$1 n=$2 /"
}
for n in ${R3_NS:-1 2}; do nproc_run timeslice $n; done
nvidia-cuda-mps-control -d
for n in 2; do nproc_run mps $n; done
echo quit | nvidia-cuda-mps-control
echo R3_DONE

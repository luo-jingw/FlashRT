#!/bin/bash
# Grid: ImageWAM executor (fp8, 10 steps, 50 ms period) + Qwen3.8-27B planner (NVFP4, 64 new tokens, back-to-back).
# Stage 0 validates the executor slimming and measures the planner alone (whole device and green partitions);
# stage 1 the executor denominators; stages 2-3 the sharing modes for the short (~560 token) and long (~8 k token)
# planner context; stage 4 MPS denominators and the bandwidth-only reference, isolated again; stage 5 cooperative
# slack scheduling (coop_gate.py) with 256-token prefill chunks, and the planner alone at that chunk length.
# usage: bash run_grid.sh [STAGE ...]   (default: all stages)
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
D=60
mkdir -p cells $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
# wait until no other user's process is on the GPU (checked before every cell)
wait_free() {
  while true; do
    busy=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      [ "$(stat -c %U /proc/$p 2>/dev/null)" != "$USER" ] && busy=1
    done
    [ $busy = 0 ] && return
    echo "WAIT_FOREIGN $(date +%T)"; sleep 30
  done
}
run() { wait_free; python colocation_bench.py "$@" 2>&1 | grep -E "hz=|Error|error|Traceback" ; }
stages=${*:-"0 1 2 3 4 5"}
for st in $stages; do
case $st in
0)
  wait_free; python exec_slim.py cells/exec_slim.json 2>&1 | grep -E "actions_equal|freed|_GB|Error|Traceback"
  for sm in 0 48 85 122; do wait_free; python planner_qwen27b.py alone $sm 5 cells/planner_alone_sm$sm.json 2>&1 \
      | grep -E "^short|^long|Error|Traceback"; done ;;
1)
  run isolated 0 $D cells/isolated_start.json
  for p in 25 50; do run greenalone$p 0 $D cells/greenalone$p.json; done ;;
2|3)
  [ $st = 2 ] && export PLANNER_CTX=short || export PLANNER_CTX=long
  for m in serial process prio green25 green50; do run $m 0 $D cells/${PLANNER_CTX}_$m.json; done
  nvidia-cuda-mps-control -d
  for p in 100 50 25; do
    if [ $p = 100 ]; then e=100; else e=$((100-p)); fi
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$e run mps$p 0 $D cells/${PLANNER_CTX}_mps$p.json
  done
  echo quit | nvidia-cuda-mps-control; sleep 3
  unset PLANNER_CTX ;;
4)
  for p in 25 50; do run greenhog$p 0 $D cells/greenhog$p.json; done
  nvidia-cuda-mps-control -d
  for e in 75 50; do CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$e run isolated 0 $D cells/mps_exec${e}_alone.json; done
  echo quit | nvidia-cuda-mps-control; sleep 3
  run isolated 0 $D cells/isolated_end.json ;;
5)
  export FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256
  wait_free; python planner_qwen27b.py alone 0 5 cells/planner_alone_sm0_chunk256.json 2>&1 | grep -E "^short|^long|Error|Traceback"
  for c in short long; do PLANNER_CTX=$c run coop 0 $D cells/${c}_coop.json; done
  unset FLASHRT_QWEN36_TQ_PREFILL_CHUNK ;;
esac
done
echo GRID_DONE $stages

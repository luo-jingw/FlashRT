#!/bin/bash
# Follow-up cells after run_grid.sh:
#   A  cooperative slack scheduling (coop_gate.py, slack-window rule) with 256-token prefill chunks, short and long;
#   B  two-process modes that do not fit with the default planner buffers: the planner worker with
#      FLASHRT_QWEN36_MAX_Q_SEQ=1024 (prefill chunk and verify/prefill scratch capped at 1024 rows), time-slicing
#      (process) and MPS without a limit (mps100), short and long; plus the planner alone at that setting.
# usage: bash run_followup.sh [PART ...]   (default: A B)
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log
D=60
mkdir -p cells $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
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
parts=${*:-"A B"}
for pt in $parts; do
case $pt in
A)
  export FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256
  for c in short long; do PLANNER_CTX=$c run coop 0 $D cells/${c}_coop.json; done
  unset FLASHRT_QWEN36_TQ_PREFILL_CHUNK ;;
B)
  export FLASHRT_QWEN36_MAX_Q_SEQ=1024
  wait_free; python planner_qwen27b.py alone 0 5 cells/planner_alone_sm0_maxq1024.json 2>&1 | grep -E "^short|^long|Error|Traceback"
  for c in short long; do PLANNER_CTX=$c run process 0 $D cells/${c}_process_maxq1024.json; done
  nvidia-cuda-mps-control -d
  for c in short long; do PLANNER_CTX=$c CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 run mps100 0 $D cells/${c}_mps100_maxq1024.json; done
  echo quit | nvidia-cuda-mps-control; sleep 3
  unset FLASHRT_QWEN36_MAX_Q_SEQ ;;
esac
done
echo FOLLOWUP_DONE $parts

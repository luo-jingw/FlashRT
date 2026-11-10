#!/bin/bash
# Follow-up cells, part 2: in-process modes (prio, green25, coop with 256-token prefill chunks) with a 0.5 ms
# interpreter switch interval (GIL handoff latency), and the executor tick time by graph capture placement.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=60
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
wait_free; python exec_capture_check.py cells/exec_capture_check.json 2>&1 | grep -E "p50|Error|Traceback"
export GIL_SWITCH_MS=0.5
for c in short long; do for m in prio green25; do PLANNER_CTX=$c run $m 0 $D cells/${c}_${m}_gil05.json; done; done
export FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256
for c in short long; do PLANNER_CTX=$c run coop 0 $D cells/${c}_coop_gil05.json; done
unset GIL_SWITCH_MS FLASHRT_QWEN36_TQ_PREFILL_CHUNK
echo FOLLOWUP2_DONE

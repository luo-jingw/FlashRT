#!/bin/bash
# Follow-up cells, part 4 (GIL-free readback + thread-local capture as in part 3): the long-context prefill as CUDA
# graphs (FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1) for prio and green25, and coop (256-token chunks) short and long.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PLANNER_READBACK_NOGIL=1 CAPTURE_THREAD_LOCAL=1
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
for m in prio green25; do FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1 PLANNER_CTX=long run $m 0 $D cells/long_${m}_nogil_pfgraph.json; done
for c in short long; do FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256 PLANNER_CTX=$c run coop 0 $D cells/${c}_coop_nogil.json; done
echo FOLLOWUP4_DONE

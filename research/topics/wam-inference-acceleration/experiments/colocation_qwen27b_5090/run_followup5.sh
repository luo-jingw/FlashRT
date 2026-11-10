#!/bin/bash
# Follow-up cells, part 5: in-process green-context partitions with both host-side fixes (GIL-free readback,
# thread-local capture, graph-captured prefill): green25 short, green50 short and long.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PLANNER_READBACK_NOGIL=1 CAPTURE_THREAD_LOCAL=1
export FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1
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
PLANNER_CTX=short run green25 0 $D cells/short_green25_nogil_pfgraph.json
for c in short long; do PLANNER_CTX=$c run green50 0 $D cells/${c}_green50_nogil_pfgraph.json; done
echo FOLLOWUP5_DONE

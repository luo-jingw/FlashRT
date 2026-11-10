#!/bin/bash
# Follow-up cells, part 3: in-process modes with the planner's per-cycle readback waiting outside the GIL
# (PLANNER_READBACK_NOGIL=1) and thread-local graph capture (CAPTURE_THREAD_LOCAL=1): prio and green25, short and long.
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
for c in short long; do for m in prio green25; do PLANNER_CTX=$c run $m 0 $D cells/${c}_${m}_nogil.json; done; done
echo FOLLOWUP3_DONE

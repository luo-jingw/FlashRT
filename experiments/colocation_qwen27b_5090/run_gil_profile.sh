#!/bin/bash
# Which code holds the GIL while the executor waits: py-spy (isolated install under tmp/pyspy) launches one in-process
# cell (prio, short context, 40 s window) and samples only the thread holding the GIL (--gil), with native frames and
# thread ids, 100 Hz. Output: collapsed stacks with sample counts (cells/gil_profile_prio_short.txt).
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PLANNER_CTX=short
PYTHONPATH=$PYTHONPATH /home/jingwu/workspace/FlashRT/tmp/pyspy/bin/py-spy record --gil --native --threads --rate 100 \
  --format raw -o cells/gil_profile_prio_short.txt -- python colocation_bench.py prio 0 40 cells/gil_profile_prio_short.json 2>&1 \
  | grep -E "hz=|Error|error|Traceback|py-spy|Samples"
echo GIL_PROFILE_DONE

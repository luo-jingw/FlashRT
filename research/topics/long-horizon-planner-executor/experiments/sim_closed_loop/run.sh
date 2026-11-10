#!/bin/bash
# usage: ./run.sh MODE OUT_DIR TASK:EPISODE [TASK:EPISODE ...]   (MODE flat | planner_eos | planner_fast)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate flashrt
source "$ROOT/tmp/env_5090.sh"
source "$ROOT/tmp/libero_env.sh"
export PYTHONPATH="$ROOT/tmp/ImageWAM:${PYTHONPATH}"
export FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1
cd "$HERE"
"$LIBERO_PY" closed_loop_libero.py "$@"

#!/usr/bin/env bash
# Jetson Thor: planner prefill and decode time against context length, under the shared GPU lock.
# usage (on Thor, FlashRT venv active): bash run_thor.sh OUT_DIR
# env: PLANNER_CKPT, PLANNER_MTP_DIR, PLANNER_PYDEPS as in ../thor_bringup/run_bringup.sh
set -u
OUT=$1
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
LOCK="$EXP/colocation_thor/with_gpu_lock.sh"
PLANNER_CKPT=${PLANNER_CKPT:-$HOME/thor_val/ckpt/qwen38-27b-nvfp4-official-5090}
PLANNER_MTP_DIR=${PLANNER_MTP_DIR:-$HOME/thor_val/ckpt/qwen38-27b-fp8-official}
PLANNER_PYDEPS=${PLANNER_PYDEPS:-$HOME/thor_val/pydeps_tf5}
{ git -C "$ROOT" log --oneline -1; head -1 /etc/nv_tegra_release; nvidia-smi; date -Is; } > "$OUT/00_state.log" 2>&1
# one process per length: the frontend sizes its buffers to the longest length it is given
for L in ${LENGTHS:-1024 2048 4096 8192 16384 32768 65536 131072}; do
  ( cd "$HERE" && \
    FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt \
    PYTHONPATH="$PLANNER_PYDEPS${PYTHONPATH:+:$PYTHONPATH}" PLANNER_CKPT="$PLANNER_CKPT" PLANNER_MTP_DIR="$PLANNER_MTP_DIR" \
    bash "$LOCK" python context_sweep.py thor "$OUT/context_sweep_thor_$L.json" "$L" ) >> "$OUT/01_sweep.log" 2>&1
done
date -Is >> "$OUT/00_state.log"
tail -10 "$OUT/01_sweep.log"

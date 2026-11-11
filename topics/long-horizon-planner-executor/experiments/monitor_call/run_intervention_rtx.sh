#!/bin/bash
# RTX 5090: the monitor-call grid on the closed-loop episodes with a non-continue decision or a failure
# (../sim_closed_loop/calls/), official Qwen3.8-27B with its own replies (the denominator of Thor row H11) and the
# LoRA-merged master with the recorded replies (the model that made those decisions in the closed loop).
# Needs the FlashRT conda env and tmp/env_5090.sh (FLASHRT_ROOT).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CALLS="$HERE/../sim_closed_loop/calls"
CK="$FLASHRT_ROOT/tmp/checkpoints"
cd "$HERE"
PLANNER_CKPT="$CK/qwen38-27b-nvfp4-official" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python monitor_call_bench.py rtx own "$HERE/monitor_call_sim_rtx_official.json" "$CALLS"/t*.jsonl
PLANNER_CKPT="$CK/qwen27-tmp-nvfp4" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python monitor_call_bench.py rtx recorded "$HERE/monitor_call_sim_rtx_master.json" "$CALLS"/t*.jsonl

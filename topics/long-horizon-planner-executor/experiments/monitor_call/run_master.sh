#!/bin/bash
# RTX 5090: the six recorded episodes (LIBERO-10 episodes 0, 43, 188; wrist 224 and 720) with the LoRA-merged master
# (the model that recorded the replies).
# Needs the FlashRT conda env and tmp/env_5090.sh (FLASHRT_ROOT).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CALLS="$HERE/../../../wam-inference-acceleration/experiments/harness_replay_5090/calls"
CK="$FLASHRT_ROOT/tmp/checkpoints"
cd "$HERE"
PLANNER_CKPT="$CK/qwen27-tmp-nvfp4" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python monitor_call_bench.py rtx recorded "$HERE/monitor_call_rtx_master.json" \
  "$CALLS"/ep0_wrist224.jsonl "$CALLS"/ep43_wrist224.jsonl "$CALLS"/ep188_wrist224.jsonl \
  "$CALLS"/ep0_wrist720.jsonl "$CALLS"/ep43_wrist720.jsonl "$CALLS"/ep188_wrist720.jsonl

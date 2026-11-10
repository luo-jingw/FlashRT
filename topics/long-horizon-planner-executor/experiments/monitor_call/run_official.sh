#!/bin/bash
# RTX 5090: ep0/ep43 wrist 224 with the official Qwen3.8-27B (the model on Thor), its own replies as reference.
# Needs the FlashRT conda env and tmp/env_5090.sh (FLASHRT_ROOT).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CALLS="$HERE/../../../wam-inference-acceleration/experiments/harness_replay_5090/calls"
CK="$FLASHRT_ROOT/tmp/checkpoints"
cd "$HERE"
PLANNER_CKPT="$CK/qwen38-27b-nvfp4-official" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python monitor_call_bench.py rtx own "$HERE/monitor_call_rtx_official.json" \
  "$CALLS"/ep0_wrist224.jsonl "$CALLS"/ep43_wrist224.jsonl

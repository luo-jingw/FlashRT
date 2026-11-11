#!/bin/bash
# RTX 5090: run-to-run repeatability of the official Qwen3.8-27B's monitor replies (the denominator of Thor row H12).
# usage: run_repeat_rtx.sh [back_to_back|after_previous] [OUT_SUFFIX]
# Needs the FlashRT conda env and tmp/env_5090.sh (FLASHRT_ROOT).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CALLS="$HERE/../../../wam-inference-acceleration/experiments/harness_replay_5090/calls"
CK="$FLASHRT_ROOT/tmp/checkpoints"
cd "$HERE"
PLANNER_CKPT="$CK/qwen38-27b-nvfp4-official" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python reply_repeat.py rtx "${1:-back_to_back}" "$HERE/reply_repeat_rtx_official${2:-}.json" \
  "$CALLS"/ep0_wrist224.jsonl "$CALLS"/ep43_wrist224.jsonl

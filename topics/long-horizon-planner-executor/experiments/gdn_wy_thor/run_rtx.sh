#!/bin/bash
# RTX 5090: linear-attention prefill variants layer by layer (the denominator of Thor row H9), official Qwen3.8-27B.
# Needs the FlashRT conda env and tmp/env_5090.sh (FLASHRT_ROOT).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CK="$FLASHRT_ROOT/tmp/checkpoints"
cd "$HERE"
PLANNER_CKPT="$CK/qwen38-27b-nvfp4-official" PLANNER_MTP_DIR="$CK/qwen38-27b-fp8-official-mtp" \
  python gdn_variant_drift.py rtx "$HERE/gdn_variant_drift_rtx.json"

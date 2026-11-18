#!/usr/bin/env bash
# Thor bring-up for the planner + executor line: record the board state, check that the planner checkpoint copied from
# the RTX 5090 host is tensor-identical to it, run the planner monitor-call smoke test, and time the pi0.5 executor.
# Every GPU step runs under the shared GPU lock (colocation_thor/with_gpu_lock.sh). Nothing here changes global state.
#
# usage (on Thor, FlashRT venv active): bash run_bringup.sh OUT_DIR
# env: PLANNER_CKPT (default $HOME/thor_val/ckpt/qwen38-27b-nvfp4-official-5090)
#      PLANNER_MTP_DIR (default $HOME/thor_val/ckpt/qwen38-27b-fp8-official)
#      PI05_CKPT (default: the lerobot/pi05_libero snapshot in the HF cache; it must carry openpi or lerobot norm stats)
#      PLANNER_PYDEPS (default $HOME/thor_val/pydeps_tf5): transformers 5.x overlay prepended to PYTHONPATH for the planner;
#        the venv's transformers 4.53 has no processor for the Qwen3.5/3.6 VL architecture
#      SKIP_HASH=1 skips the tensor-hash step
set -u
OUT=$1
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
LOCK="$EXP/colocation_thor/with_gpu_lock.sh"
CALLS="$ROOT/research/topics/wam-inference-acceleration/experiments/harness_replay_5090/calls"
PLANNER_CKPT=${PLANNER_CKPT:-$HOME/thor_val/ckpt/qwen38-27b-nvfp4-official-5090}
PLANNER_MTP_DIR=${PLANNER_MTP_DIR:-$HOME/thor_val/ckpt/qwen38-27b-fp8-official}
PLANNER_PYDEPS=${PLANNER_PYDEPS:-$HOME/thor_val/pydeps_tf5}
PI05_CKPT=${PI05_CKPT:-$(ls -d "$HOME"/.cache/huggingface/hub/models--lerobot--pi05_libero/snapshots/*/ | head -1)}

echo "== state" | tee "$OUT/00_state.log"
{ git -C "$ROOT" log --oneline -1; head -1 /etc/nv_tegra_release; nvidia-smi; date -Is; } >> "$OUT/00_state.log" 2>&1

echo "== planner checkpoint tensor hashes"
[ "${SKIP_HASH:-0}" = 1 ] || python "$EXP/planner_weights/hash_tensors.py" "$PLANNER_CKPT/model.safetensors" "$OUT/01_planner_tensors.json" \
  > "$OUT/01_hash.log" 2>&1
[ "${SKIP_HASH:-0}" = 1 ] || sha256sum "$OUT/01_planner_tensors.json" | tee -a "$OUT/01_hash.log"

echo "== planner monitor-call smoke test (wy_lt prefill)"
( cd "$EXP/monitor_call" && \
  FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt \
  PYTHONPATH="$PLANNER_PYDEPS${PYTHONPATH:+:$PYTHONPATH}" PLANNER_CKPT="$PLANNER_CKPT" PLANNER_MTP_DIR="$PLANNER_MTP_DIR" \
  bash "$LOCK" python monitor_call_bench.py thor own "$OUT/02_monitor_call.json" \
    "$CALLS/ep0_wrist224.jsonl" "$CALLS/ep43_wrist224.jsonl" ) > "$OUT/02_monitor_call.log" 2>&1
tail -5 "$OUT/02_monitor_call.log"

echo "== pi0.5 executor latency"
( cd "$ROOT" && bash "$LOCK" python examples/quickstart.py --checkpoint "$PI05_CKPT" --benchmark 50 --warmup 300 ) \
  > "$OUT/03_pi05.log" 2>&1
tail -8 "$OUT/03_pi05.log"

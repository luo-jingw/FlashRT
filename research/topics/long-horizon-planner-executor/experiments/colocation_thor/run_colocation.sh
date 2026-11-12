#!/usr/bin/env bash
# One co-location cell on the Thor: the ImageWAM executor (colocation_bench.py, mode isolated, its own process, a tick
# released every PERIOD_MS) beside the 27B planner process (planner_process.py), then colocation_align.py on the two logs.
#
# SHARING:
#   process        two processes, no MPS (the driver time-slices the two CUDA contexts)
#   mps:P:E        an MPS daemon for this cell; planner process CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P, executor E
#
# The executor's timed window starts LEAD_S after launch (its load time); the planner is launched at that moment and
# loads, warms up and plays its recordings; the executor stops at its first release after the planner exits
# (STOP_FILE), at most DURATION_S. Only ticks inside the planner's timed pass are counted by colocation_align.py.
# Uses the caller's environment for both models: CKPT_PATH, FLUX2_SRC, FLUX2_AE_MODEL_PATH, QWEN3_MODEL_SPEC (executor,
# as in scripts/imagewam_thor_validation.sh), PLANNER_CKPT, PLANNER_MTP_DIR and any FLASHRT_QWEN36_* (planner).
# Never changes machine state (no sudo, nvpmodel, jetson_clocks). Run it under with_gpu_lock.sh.
#
# usage: OUT=<dir> PERIOD_MS=533 [LEAD_S=180] [DURATION_S=1800] [ALONE=<planner jsonl>] \
#          bash run_colocation.sh TAG SHARING ARM CALLS_JSONL...
set -u
: "${OUT:?set OUT}" "${PERIOD_MS:?set PERIOD_MS}"
TAG=${1:?TAG}; SHARING=${2:?SHARING}; ARM=${3:?ARM}; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
EXEC_DIR="$(cd "$HERE/../../../wam-inference-acceleration/experiments/colocation_5090" && pwd)"
export FLASHRT_ROOT="${FLASHRT_ROOT:-$(cd "$HERE/../../../../.." && pwd)}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-$(dirname "$CKPT_PATH")/dataset_stats.json}"
export DROP_TEXT_ENCODER=1 NO_SMI=1 PERIOD_MS
LEAD_S=${LEAD_S:-180}; DURATION_S=${DURATION_S:-1800}
mkdir -p "$OUT/$TAG"
STOP="$OUT/$TAG/stop"; rm -f "$STOP"
P_PCT=""; E_PCT=""
case "$SHARING" in
  process) ;;
  mps:*:*)
    P_PCT=$(echo "$SHARING" | cut -d: -f2); E_PCT=$(echo "$SHARING" | cut -d: -f3)
    export CUDA_MPS_PIPE_DIRECTORY=$OUT/$TAG/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$OUT/$TAG/.mps/log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d ;;
  *) echo "unknown SHARING $SHARING"; exit 2 ;;
esac
echo "cell $TAG sharing=$SHARING planner_pct=${P_PCT:-none} executor_pct=${E_PCT:-none} arm=$ARM $(date -Is)"
env | grep -E '^(FLASHRT_QWEN36|CUDA_MPS)' | sort
tegrastats --interval 100 --logfile "$OUT/$TAG/tegrastats.log" & TS=$!
START_AT=$(( $(date +%s) + LEAD_S ))
( cd "$EXEC_DIR" && env ${E_PCT:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$E_PCT} START_AT=$START_AT STOP_FILE="$STOP" \
    python colocation_bench.py isolated 0 "$DURATION_S" "$OUT/$TAG/exec.json" > "$OUT/$TAG/exec.log" 2>&1 ) & EX=$!
sleep $(( START_AT - $(date +%s) > 0 ? START_AT - $(date +%s) : 0 ))
echo "executor window start $(date -Is); planner launch"
env ${P_PCT:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$P_PCT} python "$HERE/planner_process.py" thor "$ARM" \
    "$OUT/$TAG/planner.jsonl" "$@" > "$OUT/$TAG/planner.log" 2>&1
PRC=$?
sleep 5; touch "$STOP"
wait $EX; ERC=$?
kill $TS
if [ -n "$P_PCT" ]; then echo quit | nvidia-cuda-mps-control; sleep 3; fi
echo "planner rc=$PRC executor rc=$ERC $(date -Is)"
tail -n 2 "$OUT/$TAG/exec.log"
python "$HERE/colocation_align.py" "$OUT/$TAG/exec.json" "$OUT/$TAG/planner.jsonl" "$OUT/$TAG/align.json" \
    ${ALONE:+--alone "$ALONE"}

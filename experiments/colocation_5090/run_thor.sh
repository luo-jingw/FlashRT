#!/bin/bash
# Co-location grid on Jetson AGX Thor (the Thor counterpart of R1 / R4 / R5 on the RTX 5090).
# Uses the caller's environment: CKPT_PATH, FLUX2_SRC, FLUX2_AE_MODEL_PATH, QWEN3_MODEL_SPEC, PYTHONPATH as in
# scripts/imagewam_thor_validation.sh; FLASHRT_ROOT defaults to the repository root, DATASET_STATS_PATH to the
# dataset_stats.json beside CKPT_PATH, PLANNER_MODEL to $FLASHRT_ROOT/tmp/checkpoints/Qwen3-VL-2B-Instruct.
# Never changes machine state (no sudo, nvpmodel, jetson_clocks). tegrastats runs alongside every stage.
#
#   OUT=<dir> bash run_thor.sh probe                 isolated executor (full GPU, 75 % / 50 % SMs via green contexts)
#   OUT=<dir> PERIOD_MS=<ms> [MPS=1] bash run_thor.sh grid   sharing modes x planner context, then isolated again
set -u
: "${OUT:?set OUT}"
STAGE=${1:?probe or grid}
HERE="$(cd "$(dirname "$0")" && pwd)"
export FLASHRT_ROOT="${FLASHRT_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-$(dirname "$CKPT_PATH")/dataset_stats.json}"
export DROP_TEXT_ENCODER=1
D=${DURATION:-60}
mkdir -p "$OUT/cells"
cd "$HERE"
run() {  # MODE HZ TAG
  python colocation_bench.py "$1" "$2" "$D" "$OUT/cells/$3.json" > "$OUT/cells/$3.log" 2>&1
  grep -E "hz=|Error" "$OUT/cells/$3.log" | sed "s/^/$3 /"
}
tegrastats --interval 100 --logfile "$OUT/tegrastats_$STAGE.log" & TS=$!
if [ "$STAGE" = probe ]; then
  export PERIOD_MS=1000
  run isolated 0 probe_isolated
  for p in 25 50; do run greenalone$p 0 probe_greenalone$p; done
elif [ "$STAGE" = grid ]; then
  : "${PERIOD_MS:?set PERIOD_MS from the probe stage}"
  export PERIOD_MS
  run isolated 0 grid_isolated_start
  for ctx in 0 16000; do
    export PLANNER_CTX_EXTRA=$ctx
    run serial 1 ctx${ctx}_serial_hz1
    for m in process prio; do for hz in 1 0; do run $m $hz ctx${ctx}_${m}_hz$hz; done; done
    for p in 25 50; do run green$p 0 ctx${ctx}_green${p}_hz0; done
    if [ "${MPS:-0}" = 1 ]; then
      export CUDA_MPS_PIPE_DIRECTORY=$OUT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$OUT/.mps/log
      mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
      nvidia-cuda-mps-control -d
      for p in 25 50 100; do
        if [ $p = 100 ]; then e=100; else e=$((100-p)); fi
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$e run mps$p 0 ctx${ctx}_mps${p}_hz0
      done
      echo quit | nvidia-cuda-mps-control
      sleep 3
      unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    fi
  done
  unset PLANNER_CTX_EXTRA
  run isolated 0 grid_isolated_end
fi
kill $TS
echo "${STAGE}_DONE"

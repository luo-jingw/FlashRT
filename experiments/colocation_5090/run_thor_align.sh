#!/bin/bash
# Thor co-location in the short-executor configuration (ImageWAM nvfp4, 2 denoise steps, 50 / 100 ms periods), with
# denominators on the same SMs and a bandwidth-only planner stand-in. Environment as run_thor.sh. Never changes
# machine state. tegrastats runs alongside every stage.
#
#   OUT=<dir> [MPS=1] bash run_thor_align.sh probe   hog bandwidth; executor alone on all / 12 / 4 SMs (and at 50 %
#                                                     MPS threads); executor next to the bandwidth hog on 8 / 16 SMs
#   OUT=<dir> [MPS=1] bash run_thor_align.sh grid    periods 50 / 100 ms x planner context 0 / 16k x sharing mode
set -u
: "${OUT:?set OUT}"
STAGE=${1:?probe or grid}
HERE="$(cd "$(dirname "$0")" && pwd)"
export FLASHRT_ROOT="${FLASHRT_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-$(dirname "$CKPT_PATH")/dataset_stats.json}"
export DROP_TEXT_ENCODER=1 EXEC_PRECISION=nvfp4 EXEC_STEPS=2
D=${DURATION:-60}
mkdir -p "$OUT/cells"
cd "$HERE"
run() {  # MODE HZ TAG
  python colocation_bench.py "$1" "$2" "$D" "$OUT/cells/$3.json" > "$OUT/cells/$3.log" 2>&1
  grep -E "hz=|Error|Segmentation" "$OUT/cells/$3.log" | sed "s/^/$3 /"
}
mps_on() {
  export CUDA_MPS_PIPE_DIRECTORY=$OUT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$OUT/.mps/log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d
}
mps_off() {
  echo quit | nvidia-cuda-mps-control
  sleep 3
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
}
tegrastats --interval 100 --logfile "$OUT/tegrastats_align_$STAGE.log" & TS=$!
if [ "$STAGE" = probe ]; then
  export PERIOD_MS=1000
  for n in 8 16; do python bandwidth_hog.py $n 2>&1 | tail -3 | sed "s/^/hog_probe_$n /"; done
  run isolated 0 probe_isolated
  for p in 25 50; do run greenalone$p 0 probe_greenalone$p; run greenhog$p 0 probe_greenhog$p; done
  if [ "${MPS:-0}" = 1 ]; then
    mps_on
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 run isolated 0 probe_mpsalone50
    mps_off
  fi
elif [ "$STAGE" = grid ]; then
  for period in 50 100; do
    export PERIOD_MS=$period
    run isolated 0 p${period}_isolated_start
    for ctx in 0 16000; do
      export PLANNER_CTX_EXTRA=$ctx
      for m in process prio green25; do run $m 0 p${period}_ctx${ctx}_$m; done
    done
    if [ "${MPS:-0}" = 1 ]; then
      mps_on
      for ctx in 0 16000; do
        export PLANNER_CTX_EXTRA=$ctx
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 run mps50 0 p${period}_ctx${ctx}_mps50
        run mps100 0 p${period}_ctx${ctx}_mps100
      done
      mps_off
    fi
    unset PLANNER_CTX_EXTRA
    run isolated 0 p${period}_isolated_end
  done
fi
kill $TS
echo "${STAGE}_DONE"

#!/bin/bash
# Re-measure the co-location key cells with the env's pip cuBLAS 13.6.1.10 (issues.md ISSUE-103); every earlier cell
# ran under conda's 13.4.1.3. One session, outputs in cells/cublas136/.
#   cur   current tree (host-side fixes in FlashRT, no monkeypatch): executor alone, planner alone, the 8 in-process
#         configurations with the flags of run_hostfix.sh (prio / green25 / green50 short and long with the prefill
#         graph; coop short and long with eager 256-token prefill chunks), MPS 50/50 short and long as in run_grid.sh
#   orig  FlashRT before the fixes: worktree of e28b28e1 with its kernel libraries (tmp/so_backup_full_e28b28e1/,
#         flash_rt_qwen3_vl_kernels copied from the current tree), the same 8 configurations with the flags of
#         run_grid.sh stages 2, 3 and 5, plus green25 long with the research monkeypatches of run_followup3.sh
#         (PLANNER_READBACK_NOGIL=1 CAPTURE_THREAD_LOCAL=1)
#   end   executor alone again (drift bracket)
#   orig_prio_retry  orig prio short and long once more (both crashed in the first pass with a capture error)
# Every process loads site-packages/nvidia/cu13/lib/libcublas{,Lt}.so.13 (13.6.1.10); each cell logs the loaded
# libcublasLt path from /proc/<pid>/maps.
# usage: bash run_cublas136.sh [STAGE ...]   (default: cur orig end)
HERE=$(cd "$(dirname "$0")" && pwd)
WT=/home/jingwu/workspace/FlashRT/tmp/wt_orig_e28b28e1
OUT=$HERE/cells/cublas136
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=$HERE/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HERE/.mps/log
D=60
mkdir -p $OUT $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
wait_free() {
  while true; do
    busy=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      [ "$(stat -c %U /proc/$p 2>/dev/null)" != "$USER" ] && busy=1
    done
    [ $busy = 0 ] && return
    echo "WAIT_FOREIGN $(date +%T)"; sleep 30
  done
}
# bench DIR OUTNAME MODE [env ...]: one 60 s window of colocation_bench.py from DIR
bench() {
  local dir=$1 name=$2 mode=$3; shift 3
  wait_free
  echo "== $(date +%T) $name  clk $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
  (cd $dir && env "$@" python colocation_bench.py $mode 0 $D $OUT/$name.json 2>&1 | grep -E "hz=|Error|error|Traceback") &
  local job=$!
  sleep 45; for p in $(pgrep -u $USER -f "colocation_bench.py $mode 0 $D $OUT/$name.json"); do
    grep -m1 -o "/[^ ]*libcublasLt\.so[^ ]*" /proc/$p/maps 2>/dev/null | sed "s|^|   cublasLt |"; done
  wait $job
}
CUR=$HERE
ORIG=$WT/research/experiments/colocation_qwen27b_5090
ORIG_PY="PYTHONPATH=$WT:$PYTHONPATH"
stages=${*:-"cur orig end"}
for st in $stages; do
case $st in
cur)
  bench $CUR isolated_start isolated
  wait_free; (cd $CUR && python planner_qwen27b.py alone 0 5 $OUT/planner_alone_sm0.json 2>&1 | grep -E "^short|^long|Error|Traceback")
  for m in prio green25 green50; do for c in short long; do
    bench $CUR cur_${c}_$m $m PLANNER_CTX=$c FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1 EXEC_CHECK_ACTIONS=1
  done; done
  for c in short long; do
    bench $CUR cur_${c}_coop coop PLANNER_CTX=$c FLASHRT_QWEN36_TQ_PREFILL_GRAPH=0 FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256 EXEC_CHECK_ACTIONS=1
  done
  nvidia-cuda-mps-control -d
  for c in short long; do bench $CUR cur_${c}_mps50 mps50 PLANNER_CTX=$c CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50; done
  echo quit | nvidia-cuda-mps-control; sleep 3 ;;
orig)
  for m in prio green25 green50; do for c in short long; do
    bench $ORIG orig_${c}_$m $m $ORIG_PY PLANNER_CTX=$c
  done; done
  for c in short long; do bench $ORIG orig_${c}_coop coop $ORIG_PY PLANNER_CTX=$c FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256; done
  bench $ORIG orig_long_green25_patch_readback green25 $ORIG_PY PLANNER_CTX=long PLANNER_READBACK_NOGIL=1 CAPTURE_THREAD_LOCAL=1 ;;
end)
  bench $CUR isolated_end isolated ;;
orig_prio_retry)
  for c in short long; do bench $ORIG orig_${c}_prio_retry prio $ORIG_PY PLANNER_CTX=$c; done ;;
esac
done
echo CUBLAS136_DONE $stages

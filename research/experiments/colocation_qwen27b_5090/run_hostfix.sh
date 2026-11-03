#!/bin/bash
# Host-side fixes moved into FlashRT (plan "host-side decoupling for a planner and an executor in one process"):
# the research monkeypatches of colocation_bench.py are off unless a stage says otherwise; outputs go to cells/hostfix/.
#   p1  after the GIL-free cuda_read_i32_sync: planner alone (whole device), prio short with only CAPTURE_THREAD_LOCAL=1
#   p2  after thread-local capture in the frontends: planner alone, prio and green25 short and long with no monkeypatch,
#       three 60 s windows each (the third of prio short under py-spy --gil sampling)
#   p5  no monkeypatch: prio, green25, green50 short with the prefill graph, green50 long, coop short and long
# Long context runs with FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1 (short too in p5), except coop: its gate wraps the eager prefill chunks, so coop
# keeps the eager prefill with 256-token chunks (as in run_grid.sh stage 5 and run_followup4.sh).
# Every cell runs with EXEC_CHECK_ACTIONS=1 (p1 ran before the flag existed).
# usage: bash run_hostfix.sh STAGE [STAGE ...]
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PLANNER_READBACK_NOGIL=0 CAPTURE_THREAD_LOCAL=0
export EXEC_CHECK_ACTIONS=1  # fixed executor noise, per-tick action diff, per-call planner graph counts
D=60
mkdir -p cells/hostfix
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
run() { wait_free; python colocation_bench.py "$@" 2>&1 | grep -E "hz=|Error|error|Traceback" ; }
cell() {  # cell CTX MODE TAG [env ...]
  local ctx=$1 mode=$2 tag=$3; shift 3
  local pf=0; [ $ctx = long ] && pf=1
  env PLANNER_CTX=$ctx FLASHRT_QWEN36_TQ_PREFILL_GRAPH=$pf "$@" bash -c \
    "$(declare -f wait_free run); run $mode 0 $D cells/hostfix/${ctx}_${mode}_${tag}.json"
}
for st in "$@"; do
case $st in
p1)
  wait_free; python planner_qwen27b.py alone 0 5 cells/hostfix/p1_planner_alone_sm0.json 2>&1 | grep -E "^short|^long|Error|Traceback"
  cell short prio p1 CAPTURE_THREAD_LOCAL=1 ;;
p2)
  wait_free; python planner_qwen27b.py alone 0 5 cells/hostfix/p2_planner_alone_sm0.json 2>&1 | grep -E "^short|^long|Error|Traceback"
  for w in 1 2; do for ctx in short long; do for m in prio green25; do cell $ctx $m p2w$w; done; done; done
  for ctx in long; do for m in prio green25; do cell $ctx $m p2w3; done; done
  cell short green25 p2w3
  wait_free; PLANNER_CTX=short /home/jingwu/workspace/FlashRT/tmp/pyspy/bin/py-spy record --gil --native --threads --rate 100 \
    --format raw -o cells/hostfix/gil_profile_short_prio_p2w3.txt -- python colocation_bench.py prio 0 $D \
    cells/hostfix/short_prio_p2w3.json 2>&1 | grep -E "hz=|Error|error|Traceback|Samples" ;;
p5)
  for m in prio green25 green50; do cell short $m p5 FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1; done
  cell long green50 p5
  for ctx in short long; do cell $ctx coop p5 FLASHRT_QWEN36_TQ_PREFILL_GRAPH=0 FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256; done ;;
esac
done
echo HOSTFIX_DONE "$@"

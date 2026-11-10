#!/bin/bash
# Replay grid (plan "replay of the VoLoAgent Full-loop master workload with the ImageWAM executor"): the recorded calls
# of LIBERO-10 episodes 0, 43, 188 (calls/ep<E>_wrist<W>.jsonl), three episodes per process, one process per cell.
#   sync      SHARE none, wrist 224 and 720 (the harness's synchronous protocol)
#   realtime  SHARE prio, green50 (prefill graph on) and coop (eager prefill, 256-token chunks), wrist 224 and 720
# Every cell waits until no other user's process is on the GPU. Outputs: cells/<mode>_<share>_wrist<W>.json.
#   instruction  executor alone: set_prompt cost for the recorded subgoals (text encoder resident)
# usage: bash run_replay.sh STAGE [STAGE ...]     (STAGE: sync | realtime | instruction)
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p cells
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
cell() {  # cell MODE SHARE WRIST [env ...]
  local mode=$1 share=$2 w=$3; shift 3
  wait_free
  echo "== $(date +%T) $mode $share wrist$w  clk $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
  env "$@" python replay_bench.py $mode $share cells/${mode}_${share}_wrist$w.json \
    calls/ep0_wrist$w.jsonl calls/ep43_wrist$w.jsonl calls/ep188_wrist$w.jsonl 2>&1 | grep -E "jsonl|Error|Traceback"
}
for st in "$@"; do
case $st in
sync)
  for w in 224 720; do cell sync none $w FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1; done ;;
realtime)
  for w in 224 720; do
    for s in prio green50; do cell realtime $s $w FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1; done
    cell realtime coop $w FLASHRT_QWEN36_TQ_PREFILL_GRAPH=0 FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256
  done ;;
instruction)
  wait_free; python replay_bench.py instruction cells/instruction_switch.json calls/ep0_wrist224.jsonl calls/ep43_wrist224.jsonl \
    calls/ep188_wrist224.jsonl 2>&1 | grep -E "instruction switch|Error|Traceback" ;;
esac
done
echo REPLAY_DONE "$@"

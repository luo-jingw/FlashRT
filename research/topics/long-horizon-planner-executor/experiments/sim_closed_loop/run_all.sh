#!/bin/bash
# The three modes on LIBERO-10 tasks 0-9 episodes 0-1 (21 episodes with task 2 ep 2): the two episodes the executor alone fails (task 2 ep 2,
# task 8 ep 0, e8_libero_closed_loop serial10) are included. One process per mode.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/runs"
PAIRS="0:0 0:1 1:0 1:1 2:0 2:1 2:2 3:0 3:1 4:0 4:1 5:0 5:1 6:0 6:1 7:0 7:1 8:0 8:1 9:0 9:1"
for m in flat planner_eos planner_fast; do
  "$HERE/run.sh" "$m" "$OUT" $PAIRS > "$OUT/$m.log" 2>&1
done
# Instruction probes: the executor alone with an unrelated instruction, same episodes.
for m in flat_still flat_other; do
  "$HERE/run.sh" "$m" "$OUT" $PAIRS > "$OUT/$m.log" 2>&1
done

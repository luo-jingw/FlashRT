#!/bin/bash
# All tasks of task_specs.py, in order, one process per task.
# usage: run_all.sh EPISODES SEED VIDEO_EPISODES RUN_TAG
EPS=$1; SEED=$2; VEPS=$3; TAG=$4
D=/home/jingwu/workspace/FlashRT/research/topics/long-horizon-planner-executor/experiments/robotwin_instruction
ARMS=original,still,other,empty
mkdir -p $D/logs
for t in place_a2b_left place_a2b_right move_can_pot lift_pot click_bell beat_block_hammer; do
  start=$(date +%s)
  $D/run_task.sh $t $EPS $SEED $VEPS $ARMS $TAG > $D/logs/${TAG}_$t.log 2>&1
  echo "$TAG $t rc=$? $(( $(date +%s) - start ))s $(grep ARM_DONE $D/logs/${TAG}_$t.log | sed 's/.*arm=//' | tr '\n' ' ')"
done
echo "SUITE_DONE $TAG"

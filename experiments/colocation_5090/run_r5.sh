#!/bin/bash
# R5: in-process SM partition with green contexts (no MPS), the mechanism a device without MPS would use.
# Planner on P % of SMs, executor on the rest (graph recaptured on the executor's partition); continuous planner.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
export PERIOD_MS=50 DROP_TEXT_ENCODER=1
D=60
run() { python colocation_bench.py "$@" 2>&1 | grep -E "hz=|Error" ; }
mkdir -p r5
for p in 25 50; do run greenalone$p 0 $D r5/greenalone$p.json; done
for ctx in 16000 0; do
  export PLANNER_CTX_EXTRA=$ctx
  for p in 25 50; do run green$p 0 $D r5/ctx${ctx}_green${p}_hz0.json; done
done
echo R5_DONE

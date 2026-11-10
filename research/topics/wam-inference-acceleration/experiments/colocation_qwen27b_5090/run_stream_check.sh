#!/bin/bash
# ISSUE-104 check: planner frontend constructed on the default stream ("plain") and under a side stream ("side");
# output text of three calls and a digest of the language model's weight buffers (stream_check.py).
# usage: bash run_stream_check.sh TAG MODE [MODE ...]   -> smoke/stream_check_<MODE>_<TAG>.json
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
tag=$1; shift
for m in "$@"; do python stream_check.py $m smoke/stream_check_${m}_${tag}.json 2>&1 | grep -E '"text"|sha1|n_buffers|Error|Traceback'; done
echo STREAM_CHECK_DONE $tag

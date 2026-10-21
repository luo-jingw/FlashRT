cd /home/jingwu/workspace/FlashRT
source tmp/env_5090.sh; source tmp/libero_env.sh
export PYTHONPATH="/home/jingwu/workspace/FlashRT/tmp/ImageWAM:${PYTHONPATH}"
D=/tmp/claude-1002/-home-jingwu-workspace-FlashRT/e155a298-9075-4dfe-b5eb-6856aec7a685/scratchpad
$LIBERO_PY $D/libero_fewstep_closed_loop.py fp8 libero_10 5 serial10,sub_0_7_9,sub_0_9 $D/cl/l10_fp8.json > $D/cl/l10_fp8.log 2>&1
echo "exit $?" >> $D/cl/status

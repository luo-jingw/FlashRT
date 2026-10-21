cd /home/jingwu/workspace/FlashRT
source tmp/env_5090.sh; source tmp/libero_env.sh
export PYTHONPATH="/home/jingwu/workspace/FlashRT/tmp/ImageWAM:${PYTHONPATH}"
D=/tmp/claude-1002/-home-jingwu-workspace-FlashRT/e155a298-9075-4dfe-b5eb-6856aec7a685/scratchpad
for P in fp8 nvfp4 fp16; do
  $LIBERO_PY $D/libero_e2e_latency.py $P libero_10 2 $D/e2e/libero_10_$P.json > $D/e2e/libero_10_$P.log 2>&1
  echo "$P exit $?" >> $D/e2e/status
done
echo ALLDONE >> $D/e2e/status

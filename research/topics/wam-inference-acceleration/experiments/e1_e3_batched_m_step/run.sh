cd /home/jingwu/workspace/FlashRT
source tmp/env_5090.sh
D=/tmp/claude-1002/-home-jingwu-workspace-FlashRT/e155a298-9075-4dfe-b5eb-6856aec7a685/scratchpad/bm
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,memory.used --format=csv > $D/env.txt
for P in fp16 fp8 nvfp4; do
  $HOME/anaconda3/envs/flashrt/bin/python benchmarks/imagewam_batched_m_step_bench.py $P 1,2,3,4,5,6,7,8,9,10 $D/bm_$P.json > $D/bm_$P.log 2>&1
  echo "$P exit $?" >> $D/status
done
echo ALLDONE >> $D/status

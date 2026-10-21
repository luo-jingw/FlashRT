set -e
cd /home/jingwu/workspace/FlashRT
export CONDA_PREFIX=$HOME/anaconda3/envs/flashrt; source tmp/env_5090.sh
PY=$CONDA_PREFIX/bin/python; S=/tmp/claude-1002/-home-jingwu-workspace-FlashRT/e155a298-9075-4dfe-b5eb-6856aec7a685/scratchpad; F="Loading weights\|autotune\|torch_dtype\|UserWarning\|proprio_t"
$PY $S/collect_suite.py fp16 libero_10 10 -1 0 0 $S/repr2/libero_10_all.npz 2>&1 | grep -v "$F"
for s in libero_spatial libero_object libero_goal; do $PY $S/collect_suite.py fp16 $s 10 20 0 0 $S/repr2/${s}_20.npz 2>&1 | grep -v "$F"; done
$PY $S/collect_suite.py fp16 libero_10 10 -1 0 10000000 $S/repr2/libero_10_fold0_noise2.npz $S/repr2/libero10_fold0.json 2>&1 | grep -v "$F"
echo ALL_DONE

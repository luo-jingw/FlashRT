#!/bin/bash
# ISSUE-103: ImageWAM fp8 executor tick time (exec_capture_check.py, 100 ticks) with different kernel-library builds
# swapped into flash_rt/, one session, GPU otherwise idle. Builds (kernels / fa2):
#   cur      flash_rt_kernels from build_full/ (current sources incl. the GIL-free readback) / fa2 of 2026-11-03
#   premerge kernels before the upstream merge (2026-11-02) / fa2 of 2026-10-12
#   slim     slim-build kernels of 2026-10-11 / fa2 of 2026-10-12
# plus the cross pairs; the current pair runs first and last (drift bracket). The current files are restored at exit.
# Output: cells/so_compare/<kernels>_<fa2>[_end].json and a clock line per run in logs.
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate flashrt; source /home/jingwu/workspace/FlashRT/tmp/env_5090.sh
R=/home/jingwu/workspace/FlashRT; B=$R/tmp; K=flash_rt_kernels.cpython-312-x86_64-linux-gnu.so; F=flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
declare -A KS=([cur]=$B/so_backup_full_e28b28e1/flash_rt_kernels_gilfix.cpython-312-x86_64-linux-gnu.so [premerge]=$B/so_backup_premerge/$K [slim]=$B/so_backup_slim/$K)
declare -A FS=([cur]=$B/so_backup_full_e28b28e1/$F [old]=$B/so_backup_premerge/$F)
restore() { cp -p ${KS[cur]} $R/flash_rt/$K; cp -p ${FS[cur]} $R/flash_rt/$F; echo RESTORED; }
trap restore EXIT
mkdir -p cells/so_compare
for pair in cur:cur cur:old premerge:old premerge:cur slim:old slim:cur cur:cur:end; do
  IFS=: read -r k f tag <<<"$pair"
  cp -p ${KS[$k]} $R/flash_rt/$K; cp -p ${FS[$f]} $R/flash_rt/$F
  out=cells/so_compare/${k}_${f}${tag:+_$tag}.json
  echo "== $k $f $tag  clk $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
  python exec_capture_check.py $out 2>&1 | grep -E "p50|Error|Traceback|error" | head -5
  echo "   after clk $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)"
done
echo SO_COMPARE_DONE

#!/usr/bin/env bash
# ImageWAM Thor validation for the roadmap/integration branch.
#
# Runs the Thor-only checks of roadmap items 1-14 and ISSUE-020 (text_trim)
# in priority order, one log per command under $OUT, then writes
# $OUT/SUMMARY.txt with the key lines. A failing command does not stop the
# run; its exit code is appended to its log and listed in SUMMARY.txt.
# plan.md "Thor validation checklist" lists what each step decides.
#
# Before running (once):
#   1. Build this branch with GPU_ARCH=110:
#        cmake --build build -j --target flash_rt_kernels flash_rt_fp4 flashrt_imagewam_native
#        PB=$(python -c "import pybind11;print(pybind11.get_cmake_dir())")
#        cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python) -Dpybind11_DIR=$PB && cmake --build exec/build -j
#        cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python) -Dpybind11_DIR=$PB && cmake --build runtime/build -j
#   2. Copy the artifact bundle (gate fixture v1 + calibration files) and
#      check it: (cd $BUNDLE && sha256sum -c SHA256SUMS)
#   3. pip install av pandas pyarrow safetensors pillow
#
# The script never changes machine state: no sudo, no nvpmodel -m, no
# jetson_clocks. Thor is shared and runs as is (MAXN, DVFS-managed clocks);
# step 0 only records that state.
#
# Environment (required): CKPT_PATH (model.pt with dataset_stats.json and
# config.yaml beside it), FLUX2_SRC, FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH,
# QWEN3_MODEL_SPEC, DATA_ROOT (LIBERO-fastwam with libero_spatial, libero_goal,
# libero_10), PYTHONPATH=<FlashRT>:<ImageWAM>/src:$FLUX2_SRC/src, BUNDLE, OUT.
# Optional: STEPS (default "0 1 2 3 4 5 6 7 8"), PREC (default nvfp4).
#
#   OUT=$HOME/thor_val BUNDLE=$HOME/thor_bundle bash scripts/imagewam_thor_validation.sh
set -u

: "${OUT:?set OUT to an output directory}"
: "${BUNDLE:?set BUNDLE to the copied thor_bundle directory}"
for v in CKPT_PATH FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH QWEN3_MODEL_SPEC DATA_ROOT; do
  [ -n "${!v:-}" ] || { echo "missing env $v" >&2; exit 2; }
done
export AE_MODEL_PATH="${AE_MODEL_PATH:-$FLUX2_AE_MODEL_PATH}"
STEPS="${STEPS:-0 1 2 3 4 5 6 7 8}"
PREC="${PREC:-nvfp4}"
FIX="$BUNDLE/imagewam_libero_gate_v1"
CAL="$BUNDLE/imagewam_libero_calib_n64_v1.safetensors"
CAL_TRIM="$BUNDLE/imagewam_libero_calib_n64_trim_v2.safetensors"
FULL="N_TASKS=10 FRAMES=0,60 SEEDS=0,1"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

# The gate's own defaults are now the served configuration (fixture v2,
# trimming on), so every gate row below names the untrimmed reference it was
# recorded with: v1's manifest plus --no-text-trim. To gate the served default
# instead, copy the imagewam_libero_gate_v2 fixture into the bundle and run the
# same line without --manifest and --no-text-trim.
MANIFEST_V1=tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json

run() {  # run <log-name> <command...>
  local name=$1; shift
  echo "[$(date +%T)] $name: $*"
  { echo "# $*"; env "$@"; } > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "# rc=$rc" >> "$OUT/$name.log"
  [ $rc -eq 0 ] || echo "    rc=$rc"
}

want() { [[ " $STEPS " == *" $1 "* ]]; }

# 0. Provenance: commit, clocks, bundle checksums.
if want 0; then
  run 00_git git rev-parse HEAD
  run 00_clock python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state; report_jetson_clock_state()"
  run 00_bundle_sha bash -c "cd '$BUNDLE' && sha256sum -c SHA256SUMS"
fi

# 1. Full test suite: the Thor-only tests (NVFP4, SM100 CUTLASS, FA4, E0M3,
#    NVFP4 tie checks, FP8 NN-vs-TN) run here instead of skipping.
if want 1; then
  run 01_pytest python -m pytest tests/test_imagewam_*.py tests/test_jetson_clock_state.py -q -rs
fi

# 2. The served default: linear2 merge (item 4), gated-residual+AdaLN fusion
#    (item 3) and the VAE preprocessing kernel (item 2) are default-on.
if want 2; then
  run 02_e2e_default_spatial PRECISION=$PREC $FULL python benchmarks/imagewam_e2e_official_compare.py
  run 02_fusion_ab_both AB=merge_linear2,fuse_res_norm PRECISIONS=$PREC,fp16 COUNT_KERNELS=1 python benchmarks/imagewam_fusion_ab.py
  run 02_fusion_ab_linear2 AB=merge_linear2 PRECISIONS=$PREC COUNT_KERNELS=1 python benchmarks/imagewam_fusion_ab.py
  run 02_fusion_ab_resnorm AB=fuse_res_norm PRECISIONS=$PREC COUNT_KERNELS=1 python benchmarks/imagewam_fusion_ab.py
  run 02_gate_$PREC python tests/gate_imagewam_libero.py --precision $PREC --no-text-trim --manifest "$MANIFEST_V1" --fixture-dir "$FIX" --output-dir "$OUT/gate_$PREC"
  run 02_gate_fp16 python tests/gate_imagewam_libero.py --precision fp16 --no-text-trim --manifest "$MANIFEST_V1" --fixture-dir "$FIX" --output-dir "$OUT/gate_fp16"
  run 02_vae_preprocess python benchmarks/imagewam_vae_stage_bench.py --section preprocess --iters 200
fi

# 3. text_trim (ISSUE-020): accuracy vs official on three suites, speed,
#    capture cost, multi-length graph/buffer safety.
if want 3; then
  for s in libero_spatial libero_goal libero_10; do
    for t in 0 1; do
      run 03_e2e_${s}_trim$t PRECISION=$PREC SUITE=$s TEXT_TRIM=$t $FULL python benchmarks/imagewam_e2e_official_compare.py
    done
  done
  run 03_trim_bench python benchmarks/imagewam_text_trim_bench.py --precision $PREC --section all --use-fa4 off --iters 20 --rounds 5
  run 03_trim_safety TRIM_PRECISION=$PREC python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s
  run 03_trim_safety_real TRIM_PRECISION=$PREC TRIM_DIMS=real python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s
  run 03_trim_safety_e0m3 TRIM_PRECISION=e0m3_hadamard python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s
fi

# 4. Precision choice: e0m3_hadamard (item 9), NVFP4 AWQ (item 8),
#    fp8_static with real calibration (item 7), FP8 TN-vs-NN (ISSUE-001).
if want 4; then
  run 04_e0m3_thor_check python benchmarks/imagewam_e0m3_hadamard_thor_check.py
  REF="$OUT/ref_fp16.pt"
  for i in 1 2; do
    run 04_fid_nvfp4_r$i REF_CACHE=$REF PRECISION=nvfp4 python benchmarks/imagewam_precision_fidelity.py
    run 04_fid_nvfp4_awq_r$i REF_CACHE=$REF PRECISION=nvfp4 NVFP4_AWQ=1 AWQ_ALPHA=0.5 AWQ_SCOPE=adaln+down CALIBRATION="$CAL" python benchmarks/imagewam_precision_fidelity.py
  done
  run 04_fid_fp8_static REF_CACHE=$REF PRECISION=fp8_static CALIBRATION="$CAL" python benchmarks/imagewam_precision_fidelity.py
  run 04_fid_fp8_static_cutlass REF_CACHE=$REF PRECISION=fp8_static_cutlass CALIBRATION="$CAL" python benchmarks/imagewam_precision_fidelity.py
  run 04_fid_fp8_static_cutlass_placeholder REF_CACHE=$REF PRECISION=fp8_static_cutlass python benchmarks/imagewam_precision_fidelity.py
  run 04_fp8_layout ITERS=50 ROUNDS=20 python benchmarks/imagewam_fp8_layout_bench.py
  run 04_gate_fp8_static python tests/gate_imagewam_libero.py --precision fp8_static --no-text-trim --manifest "$MANIFEST_V1" --fixture-dir "$FIX" --fp8-calibration "$CAL" --iters 100 --output-dir "$OUT/gate_fp8_static"
  run 04_e2e_goal_fp8_static_trim PRECISION=fp8_static_cutlass CALIBRATION="$CAL_TRIM" SUITE=libero_goal TEXT_TRIM=1 $FULL python benchmarks/imagewam_e2e_official_compare.py
fi

# 5. FA4 (item 6), opt-in via FLASHRT_THOR_FA4=1. A log line "falling back to
#    the cuBLAS attention chain" means FA4 failed and the run measured cuBLAS.
if want 5; then
  run 05_fa4_status python -c "from flash_rt.hardware.thor import fa4_backend as f; print(f.status(), f.thor_default_enabled())"
  run 05_fa4_real_shapes python -m pytest tests/test_imagewam_fa4_backbone.py -q -s -k both_sites_real_shapes
  run 05_e2e_spatial_fa4 FLASHRT_THOR_FA4=1 PRECISION=$PREC $FULL python benchmarks/imagewam_e2e_official_compare.py
  run 05_attn_kernels python benchmarks/imagewam_attention_share_bench.py --part kernels
  run 05_attn_infer FLASHRT_THOR_FA4=1 python benchmarks/imagewam_attention_share_bench.py --part infer --precision $PREC --iters 60
  run 05_trim_bench_fa4 FLASHRT_THOR_FA4=1 python benchmarks/imagewam_text_trim_bench.py --precision $PREC --section all --use-fa4 auto --use-fa4-mot on --iters 20 --rounds 5
  run 05_trim_safety_fa4 TRIM_PRECISION=$PREC TRIM_FA4=on python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s
fi

# 6. VAE native encoder and in-graph VAE (item 5).
if want 6; then
  run 06_vae_encode python benchmarks/imagewam_vae_stage_bench.py --section encode --iters 100
  run 06_vae_infer env -u CKPT_PATH python benchmarks/imagewam_vae_stage_bench.py --section infer --precision $PREC --iters 40
  run 06_e2e_native_vae PRECISION=$PREC N_TASKS=3 FRAMES=0 VAE_ENCODER=native VAE_GRAPH=1 python benchmarks/imagewam_e2e_official_compare.py
fi

# 7. ActionDiT small-M tile tuner (item 1).
if want 7; then
  run 07_tile_kernels python benchmarks/imagewam_thor_small_m_tile_sweep.py --part kernels
  run 07_tile_infer python benchmarks/imagewam_thor_small_m_tile_sweep.py --part infer --iters 60
fi

# 8. Model-runtime ABI (item 12) and native C++ overlay (item 14) parity.
if want 8; then
  run 08_gate_abi python tests/gate_imagewam_model_runtime_export.py --precision $PREC
  run 08_gate_abi_vae_graph python tests/gate_imagewam_model_runtime_export.py --precision $PREC --vae-graph-input 224 224
  run 08_gate_native_schema python tests/gate_imagewam_native_schema_parity.py --precision $PREC
  run 08_native_tests IMAGEWAM_NATIVE_PRECISION=$PREC python -m pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q -s
  run 08_gate_native python tests/gate_imagewam_native_parity.py --precision $PREC --graph native --bench-iters 50
fi

# Summary: exit codes, pytest totals, gate verdicts, e2e and fidelity lines.
{
  echo "commit: $(git rev-parse HEAD)"
  echo "== nonzero exit codes"
  grep -l '^# rc=[1-9]' "$OUT"/*.log 2>/dev/null | xargs -r -n1 basename
  for f in "$OUT"/*.log; do
    n=$(basename "$f" .log)
    if [ "$(wc -l < "$f")" -le 6 ]; then
      lines=$(grep -v '^# ' "$f")
    else
      # imagewam_precision_fidelity.py prints its accuracy as "<col> min= median= mean=",
      # "MAE ratio", "all finite" and "peak GPU mem"; keep them with the P50 line.
      lines=$(grep -E '^(fr_vs_off|mae_fr_vs_gt|mae_off_vs_gt) |[0-9]+ passed|[0-9]+ failed|verdict|^PASS|^FAIL|P50|bit_exact|equal=|fallback|effective_config|^=== |min=.*median=|^MAE ratio|^all finite|^peak GPU mem' "$f" | tail -40)
    fi
    [ -n "$lines" ] && { echo "== $n"; echo "$lines"; }
  done
} > "$OUT/SUMMARY.txt"
echo "done: $OUT/SUMMARY.txt"

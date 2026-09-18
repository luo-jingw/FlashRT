# ISSUE-001

Status: open

Area: FP8 cuBLASLt GEMM (`fp8_gemm_descale_fp16` / `fp8_gemm_descale_f32out`, `csrc/kernels/decoder_fused.cu`), used by `Fp8Linear` and `StaticFp8Linear(use_cutlass=False)` in `flash_rt/models/imagewam/quant_linear.py`

## Observation

`fp8_gemm_descale_fp16` fails at `cublasLtMatmulAlgoGetHeuristic` with
`CUBLAS_STATUS_NOT_SUPPORTED` (status 15) on H100 (sm_90, CUDA 12.6
toolkit, torch 2.14.0+cu126) at every shape tried, from `[4,16,16]` up to
the real `[905,27648,3072]`. On the same GPU and process,
`torch._scaled_mm` with FP8 E4M3 inputs and FP16 output succeeds.

## Impact

The `fp8` and `fp8_static` precisions cannot run on sm_89 or sm_90.
FP8 accuracy work, such as real-data calibration for `fp8_static*`, can
only be checked end to end on Thor. Ada showed the same failure, and
`PROJECT.md` recorded it as an Ada cuBLASLt environment gap.

## Evidence

- The descriptor sets `CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_N` and
  `CUBLASLT_MATMUL_DESC_TRANSB = CUBLAS_OP_N`. The weight is laid out as
  column-major `(N,K)` with `ld=N`, and the activation as `(K,M)` with
  `ld=K`.
- cuBLASLt supports FP8 matmul on compute capability 8.9 and 9.0 only in
  the TN layout (A transposed, B not transposed). Blackwell lifts that
  restriction, and the same code runs on Thor (sm_110).
- `tests/test_imagewam_quant_linear.py`: on H100 the FP8 and static-FP8
  tests skip with the same status-15 message.

## Hypotheses

The NN operation layout is the only cause. With a TN descriptor over a
weight stored as `(N,K)` row-major, the heuristic would return an
algorithm on sm_89 and sm_90.

## Next Experiment

Construct a TN variant, with `TRANSA=T` and A stored as `(N,K)` row-major
(`ld=K`), and run it on H100 at the shapes above. If it works, compare
its cosine against `Fp16Linear`. Keep the NN path for Thor unless TN is
measured there to be no slower.

## Resolution

# ISSUE-002

Status: open

Area: initial action noise in the served `ImageWAMTorchFrontendThor.infer()` (`flash_rt/frontends/torch/imagewam_thor.py`)

## Observation

`infer()` fills the initial action latent with `normal_()` and then
multiplies it by `0.01`. The official `infer_action_flux2` starts from
unscaled `torch.randn`, which is N(0,1). The scaling was added in
Phase 5 (commit `f2775a5`), when the pipeline still used random weights.
It was not changed when the real checkpoint and the shift schedule were
wired in, and nothing in the repository documents a reason for it.

## Impact

The served policy samples from a narrower initial distribution than
the official model. It behaves close to a fixed-noise mode, not the
trained sampler.

## Evidence

Source: `benchmarks/imagewam_e2e_official_compare.py` on H100, fp16,
libero_spatial, 10 tasks x frames {0,60}, 20 frames in total.

| comparison | median | min |
|---|---:|---:|
| FlashRT vs official, same N(0,1) noise | 0.99840 | 0.99567 |
| FlashRT with 0.01*noise vs official | 0.99697 | 0.99301 |
| served `infer()` vs official | 0.99602 | 0.98828 |
| official seed 0 vs seed 1 | 0.99630 | 0.97154 |

Mean MAE of the 64-step chunk against ground truth:

| path | mean MAE |
|---|---:|
| official | 0.18538 |
| FlashRT, N(0,1) noise | 0.18359 |
| served `infer()` | 0.18396 |

## Hypotheses

The factor is a leftover from the random-weight stage, where it kept
activations small. It lowers cosine against the official model by about
0.0015 but leaves the open-loop error against ground truth unchanged,
and the deviation is smaller than the official model's own spread
across seeds.

## Next Experiment

Remove the factor so `infer()` matches the official sampler. Then
re-run the end-to-end comparison, and the 40-call stability check on
Thor, before shipping the change.

## Resolution

# ISSUE-003

Status: open

Area: `benchmarks/imagewam_real_checkpoint_validation.py`, block-level check with random inputs

## Observation

On H100 the backbone check gives cosine 0.993062. The ActionDiT check
gives 0.999962. On Thor, run 2026-09-14, the same checks gave 0.999927
and 0.999963.

## Impact

The served end-to-end path is not affected: actions reach cosine 0.998
against official on real frames (ISSUE-002 evidence). The block-level
backbone number, however, is no longer comparable across machines.

## Evidence

| | H100 (this run) | Thor (2026-09-14) |
|---|---|---|
| torch | 2.14.0+cu126 | 2.9.1+cu130 |
| backbone cosine | 0.993062 | 0.999927 |
| ActionDiT cosine | 0.999962 | 0.999963 |

## Hypotheses

The official bf16 path selects a different SDPA or matmul backend on
H100 with torch 2.14. The random inputs may also carry fp16 residual
magnitudes that the served path avoids with its BF16 residual.

## Next Experiment

Compare the backbone per layer on H100 with the SDPA backend pinned
(`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`) to locate the first
divergent layer.

## Resolution

# ISSUE-010

Status: open

Area: `ImageWAMTorchFrontendThor._autotune_gemm` (`flash_rt/frontends/torch/imagewam_thor.py`), `fp16` precision only

## Observation

The merged single-stream `linear1` GEMM shapes, `(a0, 3*hidden +
2*mlp_hidden, hidden)` and `(num_action, 3*action_attn_width +
2*action_mlp_hidden, action_hidden_dim)`, are not in the autotune shape
set. Every other `fp16` weight GEMM shape is, including the merged
`linear2` shapes added by roadmap item 4. With `merge_qkv_mlp` on (the
default for `fp16`), `GemmRunner.fp16_nn` runs `linear1` with the
cuBLASLt heuristic's first pick.

## Impact

`fp16` only (quantized precisions do not use `fp16_nn` for these GEMMs).
Size unknown: autotune can only match or beat the heuristic pick.

## Evidence

`_autotune_gemm`'s `shapes` set lists the split `qkv` and `mlp_in` shapes
but no `3*hidden + 2*mlp_hidden` entry; commit 4e9f7d7 added the merged
GEMM without adding its shape.

## Hypotheses

An omission in the `linear1` merge, not a deliberate choice.

## Next Experiment

Add both shapes, then A/B `fp16` `infer()` P50 on Thor with and without
them in the same process.

## Resolution

# ISSUE-011

Status: open

Area: merged single-stream `linear2` (roadmap item 4) under `fp8`, `fp8_static`, `fp8_static_cutlass`

## Observation

`merge_linear2` is on for the FP8 precisions. `Fp8Linear` quantizes its
input with one per-tensor absmax scale, and `StaticFp8Linear` with one
calibrated per-tensor scale, so the merged GEMM quantizes the attention
output and the SiLU-GLU activation with one shared scale where the split
path used one scale per half.

## Impact

If the two halves differ strongly in magnitude, the smaller half loses
FP8 resolution. None of these precisions is the shipped default
(`nvfp4`, whose per-16-element block scales make the merged and split
operands identical). No FP8 numbers exist for the merged path: FP8 GEMMs
fail on H100 (ISSUE-001).

## Evidence

`quant_linear.py`: `Fp8Linear.__call__` runs `quantize_fp8_device_fp16`
over the whole `(m, k)` input; `StaticFp8Linear.calibrate` freezes one
`act_scale`.

## Hypotheses

The effect is small next to the existing FP8 calibration gap (the
`N(0, 0.1)` placeholder activations), but it is unmeasured.

## Next Experiment

On Thor, `AB=merge_linear2 PRECISIONS=fp8,fp8_static` with `CKPT_PATH`
set (`benchmarks/imagewam_fusion_ab.py`): compare merged vs split action
cosine; if it is worse than `fp16`'s merged vs split, keep FP8 on the
split path (`dims_override={"merge_linear2": False}` as the default for
those precisions).

## Resolution

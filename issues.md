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

# ISSUE-050

Status: open

Area: NVFP4 and E0M3 weight block scales (`quantize_fp4_dynamic_sfa_fp16`, `quantize_e0m3_dynamic_sfa_fp16`), used by `Nvfp4Linear` (`flash_rt/models/imagewam/quant_linear.py`)

## Observation

The repository's block-scaled weight quantizers store `amax/6` (NVFP4)
or `amax/7` (E0M3) per 16 K values directly as UE4M3, with no
per-tensor scale. ImageWAM's weights have a median per-16 block amax of
about 0.05, so most block scales fall below 2^-6, the smallest normal
UE4M3 value.

## Impact

Below 2^-6, UE4M3 has an absolute step of 2^-9. A scale near 0.007 then
carries up to about 14% rounding error, versus at most 6.25% in the
normal range. This adds weight quantization error to the shipped
`nvfp4` tier.

## Evidence

H100, real checkpoint, 30 sampled weight tensors across backbone and
ActionDiT: fraction of blocks whose scale is subnormal is 69-100% for
NVFP4 (typically above 95%) and 84-100% for E0M3.

Simulated per-GEMM weight-only output error on real activations
(`benchmarks/imagewam_e0m3_accuracy_study.py`, 4 LIBERO frames, 180
weights pooled): `nvfp4` 0.03431, `nvfp4` with a per-tensor power-of-two
weight pre-scale 0.03299 (all 180 weights improve), `nvfp4` with MSE
scale search (`quantize_fp4_dynamic_sfa_mse_fp16`) 0.03185.

## Hypotheses

A power-of-two per-tensor pre-scale, undone exactly through the GEMM
`alpha`, removes the subnormal-scale error at zero run-time cost. The
`e0m3_hadamard` tier applies it; `nvfp4` does not.

## Next Experiment

On Thor, run `nvfp4` with the weight pre-scale and with the MSE weight
quantizer against plain `nvfp4` using
`benchmarks/imagewam_e0m3_hadamard_thor_check.py`-style fp16
comparisons, before changing the shipped tier.

## Resolution

# ISSUE-051

Status: open

Area: NVFP4 activation quantization of the backbone text-stream `txt_mlp2` input (`Nvfp4Linear`, `quantize_fp4_dynamic_sfa_fp16`)

## Observation

The text-stream MLP down-projection input in the backbone double blocks
reaches an absolute value of 5580 on real LIBERO prompts. An NVFP4
block scale for that block would be 930, above UE4M3's maximum of 448,
so the scale saturates and the block's large values clip at
6 x 448 = 2688.

## Impact

Simulated `nvfp4` output error at `txt_mlp2` is 0.0948, against 0.03-0.06
for comparable GEMMs, and it dominates the pooled per-GEMM error. The
tokens involved are text tokens whose keys and values feed every
attention that follows.

## Evidence

`benchmarks/imagewam_e0m3_accuracy_study.py` activation statistics (4
frames): `txt_mlp2` absmax 5580, fraction of saturated blocks 6e-6.
Per-GEMM error at `txt_mlp2`: `nvfp4` 0.09482; with a per-16 Hadamard
rotation (outlier spread to 1395) 0.04147; with a per-tensor
activation pre-scale 0.03350; `e0m3_hadamard` 0.03440.

## Hypotheses

A single outlier channel in the text stream, amplified by SwiGLU,
exceeds the scale range of an unscaled UE4M3 block scale.

## Next Experiment

Confirm the saturation on Thor by dumping the `txt_mlp2` input scale
bytes (0x7E = 448) from `Nvfp4Linear`'s activation scratch on one
frame.

## Resolution

# ISSUE-052

Status: open

Area: 4-bit activation quantization of the MLP down-projection inputs (backbone `mlp_down`, `img_mlp2`/`txt_mlp2`; ActionDiT `mlp_down`/`mlp2`)

## Observation

The SwiGLU outputs that feed the down projections are small and sparse.
In the backbone single blocks, 52% of per-16 activation blocks have a
subnormal NVFP4 scale (55% for E0M3) and 0.2% round to a zero scale,
which zeroes the whole block. The per-16 Hadamard rotation raises these
fractions (57%, 60%, 0.5%) because it spreads large values and lowers
the block amax.

## Impact

`bb.single.mlp_down` stays among the highest-error GEMMs in every
simulated tier (`nvfp4` 0.0863, `e0m3_hadamard` 0.0743; the attention
output projection is the other). A per-tensor activation pre-scale
computed from each call's own amax does not lower it (0.08432 to
0.08430 under `nvfp4`), because the same tensors also carry large
outliers.

## Evidence

`benchmarks/imagewam_e0m3_accuracy_study.py`, 4 LIBERO frames, activation
block statistics and per-GEMM table.

## Hypotheses

A per-row (token) activation scale, or a static per-GEMM scale
calibrated on real data (roadmap item 7), would move these blocks into
UE4M3's normal range.

## Next Experiment

Simulate a per-row power-of-two pre-scale for the down-projection
inputs in the accuracy study and measure the per-GEMM and whole-pipeline
change.

## Resolution

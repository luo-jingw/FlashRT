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

# ISSUE-030

Status: open (decision needed: served resize filter)

Area: VAE input resize in the served path
(`vae_encoder._prep_view`, and `VaePreprocessor(resize="area")`, the
frontend default `vae_resize="area"`), against the official LIBERO eval
(`ImageWAM/experiments/libero/eval_libero_single._center_crop_resize`)
and training (`torchvision.transforms.Resize([224, 224])` per camera,
`config.yaml` `processor.val_transforms`/`train_transforms`)

## Observation

The served path resizes each camera view to 224x224 with
`F.interpolate(mode="area")`. The official eval uses PIL
`Image.resize(BILINEAR)` plus a center crop (a no-op for the square
512x512 LIBERO frames). Training resizes with torchvision `Resize`
(bilinear, antialias). `benchmarks/imagewam_e2e_official_compare.py`
feeds the same PIL-resized 224x224 views to both sides by default, so its
baseline does not see this difference.

## Impact

With raw frames the served policy sees a slightly different image than
the model saw in training and in the official eval. Measured action
effect on open-loop LIBERO frames is small (below the official model's
own seed-to-seed spread); closed-loop effect is unmeasured.

## Evidence

Resize filters on real 512x512 LIBERO frames (episode 0, frame 0, both
cameras), uint8 levels: torchvision `Resize` vs PIL mean |diff| 0.07,
max 1; area vs PIL mean |diff| 0.66-0.80, max 31-34.

Same 20 raw frames (libero_spatial, 10 tasks x frames {0,60}), served
area vs `pil_bilinear` preprocessing, torch BF16 VAE on H100:

| quantity | min | median | max |
|---|---:|---:|---:|
| VAE input image cosine | 0.99887 | 0.99934 | 0.99957 |
| VAE token cosine | 0.98757 | 0.98933 | 0.99077 |
| VAE token rel_l2 | 0.136 | 0.146 | 0.158 |
| VAE token max-abs | 1.15 | 1.43 | 2.30 |

End to end, `imagewam_e2e_official_compare.py`, fp16, same 20 frames,
seeds {0,1}, the official side always PIL-resized:

| FlashRT input | `fr_vs_off` median | min | mean | mean `mae_fr_vs_gt` |
|---|---:|---:|---:|---:|
| PIL-resized 224x224 (baseline) | 0.99840 | 0.99567 | 0.99803 | 0.18359 |
| raw 512x512, `vae_resize="area"` (served) | 0.99830 | 0.99531 | 0.99806 | 0.18375 |
| raw 512x512, `vae_resize="pil_bilinear"` | 0.99840 | 0.99567 | 0.99803 | 0.18359 |

Official seed 0 vs seed 1: median 0.99630, min 0.97154. Official mean
MAE vs ground truth: 0.18538. The `pil_bilinear` row reproduces the
baseline exactly because the kernel is bit-exact to the official PIL
resize (`tests/test_imagewam_vae_preprocess.py`).

Secondary: the official eval normalizes in the model dtype (BF16 math,
`bf16(bf16(v * (2/255)) - 1)`), which differs from the served table
(float32 math, then BF16) in 127 of 256 entries by at most 0.0039 (one
BF16 ulp). The end-to-end harness normalizes in float32, which equals
the served table, so this part is not in the numbers above.

## Hypotheses

The VAE amplifies the resize-filter difference to about 1% token
cosine, and the policy is robust to it in open loop. Because training
used a bilinear antialiased resize, `pil_bilinear` is the in-distribution
choice; area averaging is a different low-pass filter.

## Next Experiment

Owner decision: make `vae_resize="pil_bilinear"` the served default
(same kernel launch count and cost; bit-exact to the official eval).
A closed-loop LIBERO success-rate A/B of `area` vs `pil_bilinear` on
Thor would settle whether the difference matters beyond open loop.

## Resolution

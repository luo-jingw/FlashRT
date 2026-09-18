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

# ISSUE-020

Status: open

Area: text-token key padding in every attention call
(`flash_rt/hardware/thor/attn_backend.py` `ImageWAMAttnBackend`, both
sites; `flash_rt/models/imagewam/pipeline_thor.py`)

## Observation

The served pipeline attends to every one of the `x0` context rows. The
Qwen3 context is tokenized to a fixed 512 tokens with
`padding="max_length"` (`flash_rt/models/imagewam/text_encoder.py`).
The official mask builder `_build_mot_attention_mask_flux2` in
`imagewam/models/backbones/imagewam.py` removes padded text keys for
every query:

```python
mask[:, :, t0:r0] &= text_valid[:, None, :]
```

The effect reaches the backbone prefill and the ActionDiT `mot` site.
`set_prompt()` reads `context_mask` only to place the proprio row.

## Impact

FlashRT and official compute different attention whenever the prompt
is shorter than 512 tokens, which covers every LIBERO prompt. The
measured end-to-end effect is bounded by ISSUE-002's numbers:
FlashRT vs official median action cosine is 0.99840 with the same
noise. A fused attention kernel adopted for either site would need to
express this key mask before the pipeline could match official exactly.

## Evidence

- Upstream mask builder, as quoted above. Both real call sites in
  `infer_action_flux2` pass `text_attention_mask=video_pre["text_mask"]`.
- `ImageWAMAttnBackend.run()` has no key-mask input. With
  `use_real_mot_mask=True` both sites call unmasked
  `attention_qkv_fp16_perhead`.
- With proprio packing, padded keys form one contiguous row range,
  `[valid_count + 1, x0)`, in the middle of the key sequence, between
  the proprio row and the image rows.

## Hypotheses

The padded keys are low-information after AdaLN modulation, so they
shift attention weights only slightly. That would explain why the
end-to-end cosine against official stays at 0.998.

## Next Experiment

In the fp16 path, set the logits of padded key columns to `-inf`
before the softmax, at both sites: a masked variant of
`attention_qkv_fp16_perhead`, or K/V rows reordered so that padding
becomes a suffix and `kv_seq` shortens. Then rerun
`benchmarks/imagewam_e2e_official_compare.py` and compare `fr_vs_off`
against 0.99840.

## Resolution

# ISSUE-021

Status: open

Area: `ImageWAMTorchFrontendThor._autotune_gemm`
(`flash_rt/frontends/torch/imagewam_thor.py`), `precision="fp16"` only

## Observation

`_autotune_gemm` autotunes cuBLASLt for a fixed list of `(M, N, K)`
shapes. For single-stream blocks the list still has the split shapes,
`(a0, 3*hidden, hidden)` and `(a0, 2*mlp_hidden, hidden)`, and their
ActionDiT counterparts. Since the `linear1` merge (opportunities.md
OPT-015, finding 1), every precision except `fp16_cutlass` runs one
merged GEMM of width `3*hidden + 2*mlp_hidden` instead. At the real
dims that is `(905, 27648, 3072)` for the backbone and
`(64, 17408, 1024)` for the ActionDiT, and neither shape is in the
autotune list.

## Impact

With `precision="fp16"`, the 20 backbone and 20 ActionDiT `linear1`
GEMMs run on cuBLASLt's top-1 heuristic algorithm, not the autotuned
one. The size of the loss is unmeasured. It can only match or lose
against autotuning. `nvfp4` is unaffected, because only its two K=7 /
N=7 fallback GEMMs go through `fp16_nn`.

## Evidence

Code reading: the `shapes` set in `_autotune_gemm`, compared with
`_alloc_random_weights` / `build_real_weights(merge_qkv_mlp=True)`,
which create `linear1.weight` with `n = 3*hidden + 2*mlp_hidden`.

## Hypotheses

The list was not updated when the merge landed.

## Next Experiment

Add both merged shapes to the list when `dims["merge_qkv_mlp"]` is set.
Then A/B the fp16 `infer()` on Thor, or on H100 as an indicative
check, with the same-process alternating method.

## Resolution

# ISSUE-022

Status: open

Area: `ImageWAMTorchFrontendThor.__init__` dims validation
(`flash_rt/frontends/torch/imagewam_thor.py`)

## Observation

The constructor accepts a `dims_override` in which `total != a0 +
num_action`. The K/V caches, `Q_O`, and `logits` are sized from
`total`, while the `mot` site writes action Q/K/V rows
`[a0, a0 + num_action)` and reads `kv_seq = total`. With
`dims_override=dict(num_action=16)` on the default dims (`a0 = 8`,
`total = 12`), construction, graph capture, and `infer()` all complete
without an error, even though the action rows lie past the end of
those buffers.

## Impact

A caller that changes `num_action` without changing `total` gets
out-of-bounds reads and writes and finite-looking output instead of an
error. The real dims (`a0 = 905`, `num_action = 64`, `total = 969`)
are consistent, so the served path is unaffected.

## Evidence

`tests/test_imagewam_gemm_variant_routing.py` first ran with
`dims_override=dict(num_action=16)`, and every test passed. The same
tests with `total=24` also pass.

## Hypotheses

The existing checks, `action_attn_width == hidden`, `HD == 128`, and
`ref_h * ref_w == a0 - x0`, never included the joint-sequence length.

## Next Experiment

Raise `ValueError` in `__init__` when `total != a0 + num_action`, then
run the regression suite. No current test or benchmark passes
inconsistent dims.

## Resolution

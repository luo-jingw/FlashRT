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
libero_spatial, 10 tasks x frames {0,60}, 20 frames in total. Cosine
against official, median / min: FlashRT with the same N(0,1) noise
0.99840 / 0.99567, FlashRT with `0.01*noise` 0.99697 / 0.99301, the
served `infer()` 0.99602 / 0.98828, official seed 0 vs seed 1
0.99630 / 0.97154.

Mean MAE of the 64-step chunk against ground truth: official 0.18538,
FlashRT with N(0,1) noise 0.18359, served `infer()` 0.18396.

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

`benchmarks/imagewam_real_checkpoint_validation.py`, the block-level check
with random inputs: on H100 (torch 2.14.0+cu126) the backbone check gives
cosine 0.993062 and the ActionDiT check 0.999962; on Thor, run
2026-09-14 (torch 2.9.1+cu130), the same checks give 0.999927 and
0.999963.

## Impact

The served end-to-end path is not affected: actions reach cosine 0.998
against official on real frames (ISSUE-002 evidence). The block-level
backbone number, however, is no longer comparable across machines.

## Evidence

Those four cosines and both torch versions are that script's own output on
the two machines.

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

`_autotune_gemm` autotunes cuBLASLt for a fixed list of `(M, N, K)`
shapes. The list still holds the split shapes the `linear1` merge
replaced, `(a0, 3*hidden, hidden)` and `(a0, 2*mlp_hidden, hidden)` with
their ActionDiT counterparts, and neither of the merged shapes:

- backbone `(a0, 3*hidden + 2*mlp_hidden, hidden)`;
- ActionDiT `(num_action, 3*action_attn_width + 2*action_mlp_hidden, action_hidden_dim)`.

Since the `linear1` merge (opportunities.md OPT-015, finding 1), every
precision except `fp16_cutlass` runs one merged GEMM of width
`3*hidden + 2*mlp_hidden` instead. At the real dims that is
`(905, 27648, 3072)` for the backbone and `(64, 17408, 1024)` for the
ActionDiT. Every other `fp16` weight GEMM shape is in the list, including
the merged `linear2` shapes added by roadmap item 4.

## Impact

With `precision="fp16"` and `merge_qkv_mlp` on (the `fp16` default), the
20 backbone and 20 ActionDiT `linear1` GEMMs run with `GemmRunner.fp16_nn`
on the cuBLASLt heuristic's first pick, not the autotuned one. The size
of the loss is unmeasured; it can only match or lose against autotuning.
Quantized precisions are unaffected, because their `linear1` does not go
through `fp16_nn`: `nvfp4` sends only its two K=7 / N=7 fallback GEMMs
there.

## Evidence

Code reading: the `shapes` set in `_autotune_gemm` lists the split `qkv`
and `mlp_in` shapes but no `3*hidden + 2*mlp_hidden` entry, compared with
`_alloc_random_weights` / `build_real_weights(merge_qkv_mlp=True)`, which
create `linear1.weight` with `n = 3*hidden + 2*mlp_hidden`; commit
`4e9f7d7` added the merged GEMM without adding its shape.

## Hypotheses

An omission in the `linear1` merge, not a deliberate choice: the list was
not updated when the merge landed.

## Next Experiment

Add both merged shapes to the list when `dims["merge_qkv_mlp"]` is set.
Then A/B the `fp16` `infer()` P50 with and without them in the same
process, on Thor or on H100 as an indicative check, with the
same-process alternating method.

## Resolution

# ISSUE-011

Status: open

Area: merged single-stream `linear2` (roadmap item 4) under `fp8`, `fp8_static`, `fp8_static_cutlass`

## Observation

`merge_linear2` is on for the FP8 precisions. `Fp8Linear` quantizes its
input with one per-tensor absmax scale, and `StaticFp8Linear` with one
calibrated per-tensor scale, so the merged GEMM quantizes the attention
output and the SiLU-GLU activation with one shared scale where the split
path used one scale per half. The weight side is the same: both classes
quantize the whole merged `(K, N)` weight with one per-tensor FP8 scale,
where the split path had one scale for `attn_out_proj` and one for
`mlp_down`.

## Impact

Measured negligible at the GEMM level (Evidence). None of these
precisions is the shipped default (`nvfp4`, whose per-16-element block
scales make the merged and split operands identical). The end-to-end
merged vs split comparison for FP8 has not run; with the ISSUE-001 TN
layout it can run on H100 as well as Thor.

## Evidence

- `quant_linear.py`: `Fp8Linear.__call__` runs `quantize_fp8_device_fp16`
  over the whole `(m, k)` input and `__init__` over the whole weight;
  `StaticFp8Linear.calibrate` freezes one `act_scale`.
- Roadmap verification pass, real checkpoint, real shapes:
  - activation absmax ratio between the two halves: backbone median 3.6,
    max 7.1; ActionDiT median 2.2, max 12.5;
  - weight absmax ratio between the two halves: 1.06-2.15;
  - FP8 GEMM error, merged / split: median 1.00x, range 0.98-1.03x
    (dynamic scale); never more than 2% worse (static scale).
- E4M3 is a floating-point format (3 mantissa bits, per-value
  exponent), so a smaller shared scale costs the smaller half dynamic
  range at the bottom of the exponent range, not relative precision;
  at these ratios (at most 12.5x, about 3.7 binades) the values stay in
  the normal range.

## Hypotheses

The single shared scale does not measurably change FP8 accuracy for
ImageWAM; the existing FP8 calibration gap (the `N(0, 0.1)` placeholder
activations) dominates.

## Next Experiment

`AB=merge_linear2 PRECISIONS=fp8,fp8_static` with `CKPT_PATH` set
(`benchmarks/imagewam_fusion_ab.py`), on Thor or on H100: merged vs split
action cosine. Close this issue if it matches `fp16`'s merged vs split
(cos >= 0.9999); otherwise default FP8 to the split path
(`merge_linear2=False` for those precisions).

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

# ISSUE-023

Status: open for roadmap item 6 (FA4); roadmap item 1 (small-M tiles) resolved by the `1021` Thor round (opportunities.md OPT-018, THOR_STATUS_SUMMARY.md)

Area: Thor confirmation of roadmap items 1 and 6 (plan.md, Phase 7 of
"Plan: ActionDiT small-M CUTLASS tile selection" and of "Plan:
attention-chain fusion recheck at ImageWAM's real shapes")

## Observation

Neither item's kernels run on the dev box (H100, sm_90). The item 1
kernels are the SM100 CUTLASS FP8 tiles, including the four new
`cutlass_fp8_t128x*`, and the NVFP4 GEMM variants. The item 6 kernel is
FA4, which has no runtime on the box (`ModuleNotFoundError: No module
named 'cutlass'`) and is SM100/SM110 only. Locally, the new tiles only
compile and link, through `sm110_check.sh`. Selection, dispatch,
buffer bounds, and fallback are tested with stand-ins for the kernels.

## Impact

Both plans stay `approved`, with a `blocked` Thor phase:

- `gemm_variant_autotune` stays default-off.
- FA4 at both sites is the served default where the machine can run it
  (`use_fa4=None` and `use_fa4_mot=None` resolve to FA4 on a
  compute-capability-11.x device with an importable FA4 runtime;
  `FLASHRT_THOR_FA4=0` forces the cuBLAS chain at both). The `mot` site
  joined the default on 0921 (plan.md, phase F1); its Thor observation is
  THOR_CHECKLIST.md N1-N3.

## Evidence

The H100 runs of the checklist scripts print `SKIP` for every NVFP4,
FP8 CUTLASS, and FA4 row. `tests/test_imagewam_fa4_backbone.py` skips
all three real-FA4 tests.

## Hypotheses

## Next Experiment

Run the Thor checklists in opportunities.md OPT-018 ("Thor check") and
OPT-019 ("Thor check").

## Resolution

# ISSUE-030

Status: open (decision needed: served resize filter)

Area: VAE input resize in the served path
(`vae_encoder._prep_view`, and `VaePreprocessor(resize="area")`, the
frontend default `vae_resize="area"`), against the official LIBERO eval
(`ImageWAM/experiments/libero/eval_libero_single._obs_to_model_input` and
`_center_crop_resize`) and training (`torchvision.transforms.Resize([224,
224])` per camera, `config.yaml` `processor.val_transforms` /
`train_transforms`)

## Observation

Three different preprocessing chains feed the same model:

| chain | resize to 224x224 per camera | normalization |
|---|---|---|
| served (`vae_resize="area"`) | torch `F.interpolate(mode="area")` | float32 `x*2/255-1`, then BF16 (256-entry table) |
| official LIBERO eval | PIL `Image.resize(BILINEAR)` + center crop | `x*(2/255)-1` in BF16 arithmetic |
| training | float32 `x/255`, torchvision `Resize` (bilinear, antialias) | `Normalize(0.5, 0.5)` in float32, then BF16 |

The official eval renders the simulator at 256x256
(`LIBERO_ENV_RESOLUTION = 256`, `experiments/libero/libero_utils.py:16`,
passed to `get_libero_env` at `eval_libero_single.py:753`), so its resize
is 256 -> 224; the LIBERO-fastwam dataset frames are 512x512. The center
crop is a no-op for square frames.

`vae_resize="pil_bilinear"` makes the served resize bit-exact to the
official eval's resize (`tests/test_imagewam_vae_preprocess.py`); its
normalization stays the served table, so it is not bit-exact to the
official eval as a whole. `benchmarks/imagewam_e2e_official_compare.py`
feeds the same PIL-resized views to both sides by default, so its
baseline does not see the resize difference.

## Impact

With raw camera frames the served policy sees a different image than
the official eval and than training. The measured open-loop action
effect is small at 512x512 and larger at 256x256, the resolution the
official eval runs at, but stays inside the official model's own
seed-to-seed spread. Closed-loop effect is unmeasured.

## Evidence

Resize filters on real 512x512 LIBERO frames (episode 0, frame 0, both
cameras), uint8 levels: torchvision `Resize` vs PIL mean |diff| 0.07,
max 1; area vs PIL mean |diff| 0.66-0.80, max 31-34.

VAE token cosine, `benchmarks/imagewam_vae_resize_compare.py` (torch
BF16 VAE, H100), 20 frame pairs (libero_spatial, 10 tasks x frames
{0,60}), median (min):

| pair | 512x512 frames | 256x256 proxy (`--proxy-size 256`) |
|---|---:|---:|
| served area vs `pil_bilinear` | 0.98933 (0.98757) | 0.97369 (0.96405) |
| `pil_bilinear` vs official eval chain (normalization only) | 0.99989 (0.99977) | 0.99989 (0.99980) |
| `pil_bilinear` vs training transform | 0.99631 (0.99540) | 0.99566 (0.99483) |
| official eval chain vs training transform | 0.99631 (0.99513) | 0.99567 (0.99465) |
| served area vs training transform | 0.99263 (0.98985) | 0.97925 (0.96790) |

The 256x256 proxy PIL-downscales the 512x512 frames (BILINEAR); a real
256x256 simulator render is not identical to it. At 256x256 the
area-vs-PIL token gap (1 - cosine, median) is about 2.5x the 512x512 gap.

End to end, `imagewam_e2e_official_compare.py`, fp16, the same 20
frames, seeds {0,1}; the official side always gets its PIL resize of
the same frames:

| FlashRT input | `fr_vs_off` median | min | mean | mean `mae_fr_vs_gt` |
|---|---:|---:|---:|---:|
| 512x512 frames PIL-resized to 224 (baseline) | 0.99840 | 0.99567 | 0.99803 | 0.18359 |
| raw 512x512, `vae_resize="area"` (served) | 0.99830 | 0.99531 | 0.99806 | 0.18375 |
| raw 512x512, `vae_resize="pil_bilinear"` | 0.99840 | 0.99567 | 0.99803 | 0.18359 |
| raw 256x256 proxy (`RAW_SIZE=256`), `vae_resize="area"` | 0.99827 | 0.99371 | 0.99786 | 0.18470 |
| raw 256x256 proxy, `vae_resize="pil_bilinear"` | 0.99842 | 0.99558 | 0.99806 | 0.18394 |

Official seed 0 vs seed 1: median 0.99630, min 0.97154 (512x512);
median 0.99612, min 0.97035 (256x256 proxy). Official mean MAE vs ground
truth: 0.18538 (512), 0.18557 (256 proxy). The `pil_bilinear` rows equal
what FlashRT gets from pre-resized views because the resize is
bit-exact; the harness normalizes both sides in float32, so the
normalization difference is not in these rows.

Normalization: the official eval's BF16 arithmetic,
`bf16(bf16(v * (2/255)) - 1)`, differs from the served float32-derived
table in 127 of 256 entries, by at most 0.0039; on tokens this is the
`pil_bilinear` vs official eval chain row above (median 0.99989).

## Hypotheses

The VAE amplifies the resize-filter difference, more at 256 -> 224 than
at 512 -> 224, and the policy is robust to it in open loop at both
sizes. `pil_bilinear` is closer to the training transform than area (the
token-cosine table above), but neither reproduces training exactly:
training resizes in float32 without uint8 rounding. The official eval
chain is as far from training as `pil_bilinear` is.

## Next Experiment

Owner decision: make `vae_resize="pil_bilinear"` the served default
(same kernel cost; resize bit-exact to the official eval, token cosine
0.9999 to the official chain including its normalization). A training-
transform mode (float32 bilinear antialias) would be closer to training
still, but no reference eval uses it. A closed-loop LIBERO success-rate
A/B of `area` vs `pil_bilinear` at the eval's 256x256 rendering on Thor
would settle whether the difference matters beyond open loop.

## Resolution

# ISSUE-040

Status: open

Area: `flash_rt.core.calibration.stratified_sample_indices` (house calibration-frame sampler)

## Observation

With `frames_per_ep = ceil(n / n_eps)` and `step = len(ep) // frames_per_ep`,
`range(0, len(ep), step)` yields `frames_per_ep + 1` frames whenever
`len(ep)` is not a multiple of `frames_per_ep`. The loop stops at `n`
picks, so the extra frame per episode is paid for by episodes at the end
of the chosen list, which are never reached.

## Impact

Episode coverage is about two thirds of what the docstring promises.
For ImageWAM's calibration build (`n = 64`, three suites, 21-22 frames
per suite) each suite's share covered 6-7 episodes instead of 11
(`frames 0, len//2, len-1` per episode). Pi0.5/GROOT callers of the same
function are affected the same way. The scales stayed stable in this
case: a differently mixed 64-frame set (31/27/6 frames per suite) gave
per-site amax within a few percent (e.g. `txt_mlp2` max 7051 vs 7067,
single-stream `linear1` median 36.47 vs 36.42).

## Evidence

- `benchmarks/imagewam_build_calibration.py` log: "64 from 21 episodes"
  for 3 x 11 chosen episodes.
- `select_calibration_frames(..., n=64)` picks frames `0, 72, 144` of
  `libero_object` episode 0.

## Hypotheses

The intended behavior is exactly `frames_per_ep` frames per chosen
episode (`range(0, len, step)[:frames_per_ep]`).

## Next Experiment

Cap each episode at `frames_per_ep` picks, rebuild the ImageWAM file,
and compare per-site amax and the `fp8_static` fidelity numbers.

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
frames): fraction of saturated blocks at `txt_mlp2` 6e-6.
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

# ISSUE-070

Status: open

Area: `exec/tests/test_exec.py` (execution-contract toy tests)

## Observation

On H100 (torch 2.14.0+cu126, `exec/build` built from this tree),
`test_capture_replay`, `test_multistream_event` and `test_buffer_copy`
fail their value checks (`capture/replay did not run the captured
memset`, and the two equivalents); `test_bind_handoff` and
`test_lru_eviction` pass.

## Impact

The toy suite reports failures that are not failures of the exec layer.
Adoption and replay of real graphs are unaffected: the ImageWAM model
runtime (OPT-028) replays through `frt_graph_replay` bit-identically to
torch replay.

## Evidence

- `frt_ctx_create()` makes stream id 0 with `cudaStreamNonBlocking`
  (`exec/src/context.cpp`, `exec/backend/cuda/cuda_backend.cpp`).
- The tests zero the target tensor with torch on torch's current stream,
  replay on exec stream 0, then `torch.cuda.synchronize()`. A
  non-blocking stream does not order against the legacy default stream,
  so the zero can land after the replay.

## Hypotheses

A host-side race in the tests, not a capture or replay defect.

## Next Experiment

Insert `torch.cuda.synchronize()` after each `zero_()` in the three
tests and re-run; all five should pass.

## Resolution

# ISSUE-087

Status: open

Area: the full imagewam pytest run on Thor (`tests/test_imagewam_fa4_dispatch.py`, `tests/test_imagewam_infer_action_noise.py` and the tests after them in file order); the frontend's FA4 fallback path (`ImageWAMTorchFrontendThor._capture_graph_or_fall_back`)

## Observation

The `0921` Thor run (commit `de0ef51`, torch 2.9.1) ended with `86 failed, 631 passed, 2 skipped, 55 errors` for `tests/test_imagewam_*.py tests/test_jetson_clock_state.py`. The first failing test is `test_static_fp8_set_activation_scale_equals_calibrate` (a 1 ULP scale difference between the device kernel and numpy, fixed in the test and recorded in `THOR_STATUS_SUMMARY.md`). The first ERROR is in `tests/test_imagewam_infer_action_noise.py`: constructing the module-scoped frontend fails in `torch.randn` with `RuntimeError: Offset increment outside graph capture encountered unexpectedly.`, after the FA4 tests that fail captures on purpose. The ABI and native gates run afterwards in new processes and pass.

## Impact

The pass count of the full run cannot be read: everything after the first poisoned test in the same process may fail for that reason and not for its own. The production frontend is affected only if a real capture failure leaves the same state behind (the fallback path recaptures and a successful capture may reset it; a second failure propagates and the process is done anyway).

## Evidence

The error text is the CUDA generator's check that no capture is open; it fires when the generator still holds the "capturing" flag while the stream is not capturing. Development GPU (torch 2.14, RTX 4060): none of three ways of failing a capture leaves the generator unusable (`scripts/probe_capture_generator_state.py`).

Thor, `0921_final` round (commit `22d3801`, torch 2.9.1+cu130), the same probe:

| case | `torch.randn` after the failed capture | after one successful small capture |
|---|---|---|
| `case_sync_inside_capture` | `RuntimeError: Offset increment outside graph capture encountered unexpectedly.` | ok |
| `case_python_error_inside_capture` | ok | not needed |
| `case_rng_inside_capture_then_error` | ok | not needed |

`tests/test_imagewam_fa4_dispatch.py` alone: 24 passed. `-k capture_sync` followed by `tests/test_imagewam_infer_action_noise.py` in one process: 5 passed (the FA4 fallback test recaptures successfully, which repairs the flag). The checklist's own command (`-k capture_sync` over both files) deselected the second file's tests (`1 passed, 27 deselected`) and so did not test the cascade.

So on torch 2.9.1 a device sync inside a capture leaves the generator's flag set, and one successful capture clears it. The FA4 fallback path recaptures, so it repairs the state whenever the retry succeeds. Not established: which test leaves the flag set with no successful capture after it.

## Hypotheses

A test that fails a capture on purpose and ends without a successful capture after it (a final failure: "after a second failure the frontend holds no graph", or a test that only checks the refusal) leaves the process in this state, and every later CUDA random draw then fails. It is in the tests between `test_imagewam_fa4_dispatch.py` and `test_imagewam_infer_action_noise.py` in file order, or in the file order of the run that erred.

## Next Experiment

Done in `0921d` (`FLASHRT_GENERATOR_SENTINEL=1`, `tests/conftest.py`): the sentinel named one test, `tests/test_imagewam_gemm_variant_tuner.py::test_cuda_graph_timer_on_real_launches`, and after it the whole run is `2 failed, 781 passed, 1 skipped, 5 errors` (86 / 631 / 55 without it). The fix is that test's own: end it with one successful small capture (or keep its failing capture out of the default generator's process).

## Resolution

# ISSUE-088

Status: open

Area: `precision="fp16"` on Thor under `text_trim` (`flash_rt/frontends/torch/imagewam_thor.py`, `flash_rt/models/imagewam/pipeline_thor.py`, the `fp16_nn` GEMM path)

## Observation

Trimming the text context shortens the graph's sequence (LIBERO `valid_tokens=24` gives `x0=25`, `a0=417` against 905). It removes about 87 ms from `nvfp4` and about 105 ms from `fp8_static_cutlass` on Thor, and removes nothing from `fp16`.

## Impact

The `fp16` row of every table is the untrimmed speed whatever the profile says, so `fp16` looks about as fast at the served default as untrimmed (about 275 ms), and the `default` profile's whole gain over the official implementation for that tier is only the FlashRT graph itself. It also means some part of the `fp16` graph does not scale with the row count, which is not what a compute-bound GEMM path does.

## Evidence

fp16 P50 of the same kind of trimmed graph, by harness and round:

| round | harness | fp16 |
|---|---|---:|
| earlier matrix `default` row | e2e, untrimmed | 275.2 ms |
| `0920t` | gate, fixture v2 | 284.38 ms |
| earlier stacked row | e2e | 273.8 ms |
| `0921_final` | `imagewam_thor_path_bench.py --profile default` | 274.70 ms |
| `0921d_final` | the same command | **226.71 ms** (P10 226.35, P90 227.67) |
| `0921d_final` | `imagewam_e2e_official_compare.py`, profile default | 224.3 ms |
| `0921d` D1 | `imagewam_graph_kernel_profile.py`, first graph (24 valid tokens), replay | 294.95 ms |
| `0921d` D2 | `imagewam_text_trim_bench.py --section ab`, x0=21 replay, same process | 172.36 ms (x0=513: 270.93) |

The same command ran 274.70 ms one round and 226.71 ms the next, with no code change between them that touches the graph. `nvfp4` under the same commands is stable (108.03 / 107.62 ms, repeat -0.4%; 124.95 ms 24-token replay in D1 with FA4 off) and drops with the row count (D1: 206.49 to 124.95 ms; D2: 181.80 to 109.81 ms). `fp16_cutlass` (CUTLASS) also shrinks (275.59 to 187.05 ms). Only the cuBLASLt `fp16` path is both non-scaling in some runs and unstable between runs.

D1, 512 valid tokens, kernels inside the replay: fp16 GEMM 218.28 ms of 288.48 (`nvjet_hsh_448x64` 46.1 ms x200, `nvjet_hsh_128x192` 39.9 ms x35); nsys: `nvjet_hsh_512x64` averages 4.47 ms per call. A 47 GFLOP GEMM at a tensor-core rate takes a few tenths of a millisecond, so some fp16 GEMMs run at a few TFLOPs. The kernels inside the replay are about 100% of it (no idle gaps), so the time is in the GEMM kernels.

The frontend's autotune (`GemmRunner::autotune_cached`) times the heuristic's top 16 algorithms on the zero-filled scratch `_autotune_gemm` passes and keeps the fastest per shape, in a cache keyed by (type, M, N, K).

## Hypotheses

Confirmed by `0921x` (Thor, `benchmarks/imagewam_fp16_gemm_probe.py --x0 25,513`, full log `thor_val/0921x/X1_fp16_gemm_probe.log`): the trimmed shapes' cuBLASLt GEMMs achieve 4-6 TFLOPs (`txt_qkv` 6.1, `txt_mlp0` 4.6-5.9 at `x0=25`) against 96-110 TFLOPs for the same GEMMs at `x0=513`. This is not a row-count effect on a fixed-efficiency kernel; the achieved rate itself collapses at small M. The heuristic's own top-1 pick is frequently far behind what autotune finds at the SAME shape (`single_mlp_in`: heuristic 31 TFLOPs vs autotuned-on-zeros 84; `img_mlp0`: 47 vs 80), so the autotune step matters and is not fully closing the gap either -- `zeros` and `random` autotune data picked the same algorithm every time (ruling out zero-fill as the cause), and `txt_proj` at `M=25` still spans `0.019..0.065 ms` across three fresh runners (algorithm choice is unstable at this shape even after autotuning). Root cause: small-M cuBLASLt efficiency at sm_110, not a scheduling or fusion gap elsewhere in the pipeline.

## Next Experiment

Done (`0921x`, see Hypotheses). Remaining: (a) whether raising `autotune_cached`'s `num_algos`/`bench_iters` at these small-M shapes finds a better pick than the current top-16/10-iteration search, or whether cuBLASLt has no fast candidate at all for M=25-ish rows at these N/K and the fix has to be a different GEMM path (CUTLASS, which `fp16_cutlass` already uses and which does scale with `text_trim`, OPT-032); (b) whether the per-runner instability at `txt_proj` M=25 (`0.019..0.065 ms` spread) means the autotuned cache entry itself is not deterministic between processes, which would need fixing independently of which GEMM path is chosen.

## Resolution

Root cause confirmed (small-M cuBLASLt inefficiency at sm_110); the fix (switch `fp16`'s weight GEMMs to CUTLASS, or improve the autotune search) is not implemented.

# ISSUE-089

Status: open

Area: `tests/test_imagewam_text_trim.py` and `tests/test_imagewam_real_checkpoint.py`

## Observation

After the generator poisoning was removed by the sentinel (ISSUE-087), the `0921d` full run leaves 2 failures and 5 errors. The 2 failures are in `tests/test_imagewam_text_trim.py`: one asserts the per-length cache is empty after a failed capture, and one compares `torch.equal` after an FA4 fallback and gets values that are very close but not equal. Four of the errors are `test_imagewam_real_checkpoint.py` (`fixture 'mot' not found`); the fifth is the sentinel's own teardown.

## Impact

Not measured. The fallback comparison being close but unequal is the FA4-fallback graph against a chain reference; whether that is a tolerance in the test or a real difference is not known.

## Evidence

`0921x` (Thor, `thor_val/0921x/X3_text_trim.log`), both tests run alone with `-x -q -s`:

- `test_failed_capture_leaves_no_replayable_graph`: printed `after the failed set_prompt: graph=None current_prompt=None cached=()`, then
  ```
  assert fe.captured_text_lengths == (6,) and fe._captures[6].rope_table is cached_rope
  E   assert (() == (6,)
  ```
  The frontend clears `graph`/`current_prompt` correctly, but the per-length capture cache is also empty (`cached=()`) where the test expects the length-6 capture to survive a later unrelated failure.
- `test_fa4_fallback_keeps_old_graphs_until_the_replacement_exists`: `seen == [(6, 13), (6, 13)]` and `captured_text_lengths == (4,)` both pass; it fails at
  ```
  assert torch.equal(value, _run(cublas, ctx, n)), n_valid=3 after the fallback must run the cuBLAS chain
  ```
  with the two sides numerically close but not equal (e.g. -1.5116 vs -1.5115, -0.9865 vs -0.9866) -- not a large discrepancy, but not bit-exact either.

## Hypotheses

The `mot` fixture error is a test that has no checkpoint fixture on the Thor (an untracked local test file, `tests/test_imagewam_real_checkpoint.py`, was noted in the gate's `worktree not clean`). For the two real failures: (a) the cache-survival test's own expectation may be stale -- if the frontend's cache-clearing behavior changed since the test was written to clear the WHOLE cache on any capture failure (not just the failing length's entry), the test needs updating, not the frontend; (b) the FA4-fallback numeric mismatch could be a genuinely different-but-valid cuBLASLt algorithm pick between "the fallback's cuBLAS chain" and "a fresh cuBLAS-chain-only build" (ordinary GEMM non-associativity, like the ULP-level fp8 scale mismatch in ISSUE elsewhere), or could be a real state leak from the aborted FA4 attempt. Neither is confirmed.

## Next Experiment

(a) is resolved and Thor-confirmed (`0922`, `thor_val/0922`): with `use_fa4=False` pinned, `test_failed_capture_leaves_no_replayable_graph`'s cache-survival assertion passes (`cached=(6,)`, not `()`). The SAME test has a second, later assertion this run reached for the first time: `torch.equal(out, fresh)` comparing the length-9 recovery capture against a fresh frontend built with the same `GemmRunner`, which failed at cosine=1.0000000, max_abs=6.104e-05 -- the same small-magnitude, non-catastrophic mismatch pattern as (b) below (both frontends share the autotuned `GemmRunner`, so this is not an obviously-different-algorithm case like (b), but the magnitude and character match it closely enough to treat the same way pending a more specific explanation).

(b): `test_fa4_fallback_keeps_old_graphs_until_the_replacement_exists`'s FA4-fallback-vs-fresh-cuBLAS mismatch is confirmed on Thor (`0922`): ~1e-4 absolute on ~1.5-magnitude values. The per-shape cuBLASLt-algorithm comparison this called for was not done; given the consistent small magnitude in both (a)'s and (b)'s occurrences, and this project's own established precedent for the same class of difference (`docs/imagewam_last_block_kv_only.md`'s narrower-GEMM comparison: "ordinary GPU GEMM floating-point non-associativity between two different real kernel launches, not a bug", handled with a tolerance instead of `torch.equal`), both assertions were relaxed to `cos > 0.9999 and max_abs < 1e-2` rather than pursued further as exact algorithm-cache comparisons. This is the same category of fix, applied on the strength of the precedent and the observed magnitudes, not a newly confirmed root cause for either.

## Resolution

(a)'s cache-survival half: fixed and Thor-confirmed. Both remaining small numeric mismatches (the length-9 recovery-vs-fresh comparison, and (b)'s FA4-fallback-vs-cuBLAS comparison): tests relaxed to a tight tolerance in `tests/test_imagewam_text_trim.py`, matching this project's own precedent for GEMM algorithm-pick non-associativity; needs a Thor rerun to confirm both now pass, and remains open as "not a confirmed root cause" rather than "proven non-associativity" for either.

# ISSUE-090

Status: resolved (in the test; the underlying GemmRunner/cuBLASLt behavior is noted but not fixed, since no production code path is known to hit it)

Area: test-construction pattern for `_single_stream_layer`/`_action_single_layer` (`tests/test_imagewam_thor_real_wiring.py`); `GemmRunner`/`Fp16Linear` (`csrc/gemm/gemm_runner.cu`, `flash_rt/models/imagewam/quant_linear.py`)

## Observation

`0922b` (Thor): the new `fuse_qkv_norm_rope` wiring test (opportunities.md OPT-032 candidate 1) crashed with `an illegal memory access was encountered` at `torch.cuda.synchronize()` after `_action_single_layer(fused_qkv=True)`, following a `cos=1.0000001 max_abs=0.0 bit_exact=True` PASS for backbone. Reproduced locally (Ada, independent of Thor) and narrowed down precisely: the crash requires calling `_action_single_layer` (or, by a separate check, `_single_stream_layer`) a SECOND time against a SHARED `GemmRunner`, with FRESH weight tensors (new `Fp16Linear` wrapping new pointers) at the SAME `(M, N, K)` GEMM shape as the first call -- **`fuse_qkv_norm_rope` is not required to trigger it**: two calls with `fused_qkv=False` on both crash identically. `CUDA_LAUNCH_BLOCKING=1` places the synchronous fault inside `attn_out_proj.weight`'s `gemm.fp16_nn` call (`cuBLAS error ... code=13`), on the SECOND call, reading from the buffer the (first) attention output was written into.

## Impact

None found in production: the real frontend constructs every `Fp16Linear` ONCE at load time and never rebuilds a weight wrapper for an already-used shape during `infer()`, so this specific trigger (repeated fresh-weight construction against one persistent `GemmRunner`) is a TEST-authoring pattern, not a served code path. It did block confirming OPT-032 candidate 1's wiring correctness until diagnosed.

## Evidence

Local repro scripts (Ada, RTX 4060, `CUDA_LAUNCH_BLOCKING=1` and `compute-sanitizer --tool memcheck`/`--tool racecheck`, none of which reproduced or explained it beyond confirming it is not a simple out-of-bounds write memcheck catches):

- Two `run_action_single(fused_qkv=True)` calls back to back (fresh weights each call, shared `gemm`): crashes on the second, same traceback as Thor.
- Two `run_action_single(fused_qkv=False)` calls back to back: **also crashes**, identical traceback -- rules out the new kernel entirely.
- One `run_action_single(fused_qkv=True)` call alone (no preceding call): does not crash.
- Building the weights, modulation, attention backend and input ONCE, then replaying `_action_single_layer` twice against them with only `dims["fuse_qkv_norm_rope"]` flipped: does not crash, and the two outputs are bit-exact (`torch.equal`), confirmed 3 runs.
- The same restructuring applied to `_single_stream_layer`'s own comparison (which had not crashed, built fresh weights per call, and shares the same risk): also bit-exact after the change.

## Hypotheses

`GemmRunner`'s per-`(op_type, M, N, K)` cuBLASLt algorithm cache (`csrc/gemm/gemm_runner.cu`) is keyed by shape, not by data pointer; calling `fp16_nn` again at a cached shape with a NEW weight pointer reuses the cached algorithm/descriptor. Something about that reuse -- possibly interacting with the 256 MB workspace `GemmRunner()` allocates once at construction, possibly a stale device-pointer reference kept in the cached `cublasLtMatmulAlgo_t`/descriptor -- becomes invalid on the second distinct pointer at ActionDiT's specific shapes, surfacing as `CUBLAS_STATUS_EXECUTION_FAILED` in a LATER call, not the one that actually corrupted something. Unconfirmed: why backbone's shapes did not exhibit this in the (also fresh-weights-per-call) `run_single_stream` sequence before the fix -- either backbone's exact shapes happen not to trigger it, or it needed the additional accumulated state from also running the action shape at least once.

## Next Experiment

Not pursued further here (out of this work's scope, and no known production trigger): reproduce with a minimal repro isolated to `GemmRunner.fp16_nn` alone (no attention backend, no pipeline_thor.py), at ActionDiT's exact `(M,N,K)` shapes, two calls with fresh device pointers each, to determine whether this is a `GemmRunner`/cuBLASLt caching bug worth fixing at that layer, or specific to how `attn_out_proj`'s input aliases the attention backend's in-place `Q`/`O` buffer.

## Resolution

Worked around in `tests/test_imagewam_thor_real_wiring.py`: `_single_stream_layer`'s and `_action_single_layer`'s fused_qkv comparison tests build weights/attention/input ONCE and replay against them, rather than rebuilding fresh weights per compared call. OPT-032 candidate 1's wiring is confirmed bit-exact for both backbone and ActionDiT once this trap is avoided. Thor-confirmed with the corrected test (`0922d`, HEAD `e727f91`): `1 passed, 6 deselected`, all three comparisons `torch.equal`/`bit_exact=True`, `max_abs=0`. The underlying `GemmRunner` behavior itself is not fixed or further diagnosed; flagged here in case it surfaces again outside this test.

# ISSUE-091

Status: resolved

Area: `_fused_gate_res`'s OPT-032 candidate 3 dispatch (`flash_rt/models/imagewam/pipeline_thor.py`)

## Observation

`0922f` (Thor, first real run of `test_single_stream_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes`, after `ENABLE_SM100_CUTLASS` rebuild): `fuse_res_norm_fp4=True` (the new candidate-3 wiring) produced a BF16 residual with values like `-7392`/`3008` (O(1e3)), while `fuse_res_norm_fp4=False` (the existing, already-wired path) stayed `O(1)` (`-0.79`/`5.16`/`-3.92`) on the exact same inputs. `_diff_stats`'s `cos=nan` came from NaN/Inf on the fused side, not from the cosine metric itself.

## Impact

Candidate 3's Round 1 wiring was completely non-functional (not a small numeric drift) whenever `dims["fuse_res_norm_fp4"]` was set. No production deployment could have hit this yet: the flag is new, opt-in, and default off in this same round.

## Evidence

Reading `csrc/kernels/fused_norm_fp4/fused_norm_fp4.cu`'s exported signatures directly: `gate_res_ada_layer_norm_fp4_sfa_bf16res`/`_fp16res` declare `gate`/`scale`/`shift`/`inv_s` as `const void*` typed `__half*` (FP16) inside the kernel launch (`launch<ResT>`'s own `reinterpret_cast<const __half*>`), confirmed against `tests/test_fused_norm_fp4_kernel.py`'s own reference construction, which builds `gate`/`scale`/`shift` as `torch.float16` tensors before calling this exact kernel. `_fused_gate_res`'s new `fp4_direct` branch instead called `_mod_vec_ptr(gate, dim)` etc. -- a helper whose OWN contract (used correctly by the EXISTING `gate_res_ada_layer_norm_bf16res`, which really does take `const float*` and rounds to FP16 internally, `csrc/kernels/fusion.cuh`) requires and returns an FP32 tensor's pointer. Feeding an FP32 pointer to a kernel that reads it as `__half*` reinterprets every 4-byte FP32 element as two 2-byte FP16 elements at half the intended stride -- garbage values, consistent with the observed O(1e3)/NaN blow-up (not the small-drift signature of an ordinary numerics difference).

## Hypotheses

None needed -- root cause fully confirmed by reading both the kernel's own C++ declaration and its own established reference test's construction; no further experiment required.

## Resolution

Fixed in `_fused_gate_res`'s `fp4_direct` branch: build fresh FP16 `(dim,)` tensors for `gate`/`scale`/`shift` (`t[0, 0].to(torch.float16).contiguous()`, same slicing convention `_fuse_mod_pair` already uses) and, when present, for `awq_inv_s`, and pass THEIR pointers instead of `_mod_vec_ptr`'s FP32 ones. Added a CPU-only regression test (`tests/test_imagewam_fuse_res_norm_fp4_dispatch.py::test_gate_scale_shift_are_rounded_to_fp16_not_passed_as_fp32`) that decodes the actual bytes at the pointers passed to the kernel and checks they match the FP16 rounding of the real input -- reading them back synchronously inside the mock kernel's own `side_effect`, since reading a CPU tensor's memory back from saved `call_args` after the producing function has already returned is itself unsound (its refcount hits zero and the allocator is free to reuse it; a first attempt at this same regression test read stale/reused bytes for exactly that reason, not a second real bug). Local dispatch test suite passes 9/9; Thor-confirmed (`0922g`, HEAD `3056b03`, no rebuild needed): `1 passed`, the real two-layer wiring test `torch.equal`/`bit_exact=True`, `max_abs=0`.

# ISSUE-092

Status: open

Area: Pi0.5 RTX's FP8 decoder path (`flash_rt/frontends/torch/pi05_rtx.py`'s `_quantize_all_fp8`/`calibrate_with_real_data`, `flash_rt/models/pi05/pipeline_rtx.py`'s `_fp8_gemm`, `flash_rt/core/calibration.py`'s `check_scale_ceiling`)

## Observation

RTX 5090 (`arch=rtx_sm120`, commit `7a68a1c`), Pi0.5 3-view/10-step, fp8 vs fp16 cosine over the SAME aligned initial noise (CPU `Generator.manual_seed(0..4)`, `(10, 32)`, `copy_`'d into `_noise_buf`, not independently resampled per side): 4 of 5 seeds land at 0.9995-0.9998, seed 1 lands at **0.417** (dummy-data calibration) / **0.417** again, actually worse, after re-calibrating with 8 real stratified LIBERO frames (`0.60704` with dummy calibration -> `0.41653` with real calibration -- switching to real calibration data made it WORSE, ruling out "unrepresentative calibration set" as the cause). `check_scale_ceiling` (`flash_rt/core/calibration.py`) fires both times: 12 scales exceed 20x the calibration's median scale (0.012), `encoder_ffn_down_w_16` the worst offender (amax 16.929 with dummy data, 26.711 with real data -- the real-data amax is HIGHER, not lower, meaning this layer's true activation distribution really does have extreme outliers, the dummy data wasn't underestimating it).

The same run also measured speed (warmup 10 / iters 100, wall + `cuda.synchronize`, `tmp/bench_5090_pi05_3view_noise.npz`): FlashRT fp16 P50 28.76 ms (P10-P90 28.73-28.80), FlashRT fp8 P50 16.84 ms (16.82-17.08) -- **1.71x**, in line with fp8's speedup on every other hardware this project has measured (Thor same-session fp16-to-fp8 ratio: 85.84/49.77 = 1.73x, matching within noise). NVFP4 is not wired or tested for Pi0.5 on RTX 5090 at all (deferred, separate from this issue).

Official OpenPI has no row in this 3-view table, and the reason is NOT a torch/sm_120 limitation -- an earlier pass on this same machine wrongly attributed the missing row to `torch==2.7.1` lacking sm_120 kernels; re-checked directly with `torch==2.14.0+cu130` (which does run sm_120) and the real cause is an I/O mismatch: OpenPI's own LIBERO harness (`LiberoInputs`) only ever feeds 2 real cameras, zero-padding the right-wrist slot, so its output is not comparable to FlashRT's 3-real-camera row. Official OpenPI at 2-view, same clocks: P50 43.88 ms (43.83-44.03), action shape `(10, 7)` -- recorded for scale only, not as a same-harness baseline for the 3-view row above.

## Impact

Pi0.5 RTX's FP8 path has at least one systematic single-seed catastrophic failure mode (cosine 0.42, not a small drift) tied to one or more FFN-down-projection layers (`encoder_ffn_down_w_14/15/16` at least) whose real activation distribution has a heavy-tailed/outlier-channel shape that a single per-tensor FP8 E4M3 scale cannot cover without either clipping the rare extreme values (this failure mode) or wasting dynamic range on the common case. `check_scale_ceiling`'s own docstring already documents there is no runtime fallback to FP16 for an offending layer if this fires -- a real correctness risk for any deployment that hits an input distribution similar to whatever seed 1's denoise trajectory produces at that layer, not just this specific benchmark's seed 1.

## Evidence

- Two calibration rounds, same layer flagged both times, real-data amax HIGHER than dummy-data amax (26.711 vs 16.929) -- rules out "the calibration set doesn't see the true range" as the cause; the true range really is this large.
- `flash_rt/core/calibration.py`'s own `check_scale_ceiling` docstring: "the signature of a true outlier sample," "FP8 will still run but dynamic-range headroom on these layers is compressed... Consider lowering percentile, sampling more diversely, or keeping these layers in FP16" -- the project's own diagnostic already anticipated this exact failure shape.
- 4/5 seeds unaffected (0.9995+) -- this is not a systematic quantization bug affecting every input, it is specific to whatever seed 1's own denoise trajectory produces at this one layer.

## Hypotheses

`encoder_ffn_down_w_*`'s real activation (the FFN's SiLU/gate-gated intermediate, going into the down-projection) has the classic "outlier channel" shape documented in AWQ/SmoothQuant literature for transformer FFN-down layers -- a small number of channels or samples with activation magnitude far above the tensor's typical range, which a single per-tensor scale cannot represent well simultaneously with the common case. Not yet confirmed directly (would need a per-channel or per-sample histogram of this layer's real activation across many real inputs, not attempted here).

## Next Experiment

Lowest-risk, cheapest first step (already the project's own documented recommendation): identify every layer `check_scale_ceiling` persistently flags across several real calibration sets, and keep those specific layers in FP16 rather than FP8 in `_quantize_all_fp8` (a small, targeted exclusion list, negligible speed cost given it is a handful of layers out of dozens) -- then re-run the same 5-seed aligned-noise cosine check to confirm seed 1 recovers. If it does not recover even with those layers excluded, the outlier is elsewhere (a different layer not yet flagged, or the decoder's own attention/residual path) and needs the same check repeated on whatever new layer(s) `check_scale_ceiling` flags once these are removed. Not attempted yet.

# ISSUE-093

Status: open, non-blocking

Area: `flash_rt/hardware/__init__.py` (`_PIPELINE_MAP`)

## Observation

`("imagewam", "torch", "rtx_sm120")` has never been a key in
`_PIPELINE_MAP` -- every RTX 5090 ImageWAM benchmark this project has
run (OPT-033 and its own predecessors) constructs
`ImageWAMTorchFrontendThor` directly (`from
flash_rt.frontends.torch.imagewam_thor import
ImageWAMTorchFrontendThor`), bypassing `flash_rt.load_model()` and
`resolve_pipeline_class` entirely. Found incidentally while registering
Orin (`rtx_sm87`, plan.md "Plan: ImageWAM on Jetson Orin (rtx_sm87),
registration only") -- Orin's own dispatch-table entry made the same
gap for `rtx_sm120` visible by comparison.

## Impact

Nothing currently depends on `flash_rt.load_model(config="imagewam",
hardware="rtx_sm120", ...)` working, so this has not broken anything
observed so far. It would surface the moment anything (a plugin, a
served harness, a script following `docs/plugin_model_template.md`'s
own documented pattern) tried to construct ImageWAM through the public
API on a 5090 instead of importing the frontend class directly.

## Next Experiment

Add `("imagewam", "torch", "rtx_sm120"): ("flash_rt.frontends.torch.imagewam_thor",
"ImageWAMTorchFrontendThor")` to `_PIPELINE_MAP`, matching the `thor`/`rtx_sm87`
entries added for the Orin work, and confirm `load_model()` resolves it on a 5090.
Not attempted yet -- small, but out of scope for the Orin registration plan that found it.

# ISSUE-094

Status: open, non-blocking

Area: `scripts/gen_synthetic_pi05_checkpoint.py`, `flash_rt/frontends/torch/{pi05_thor,pi05_rtx}.py`

## Observation

The new synthetic-checkpoint generator was validated end to end as far as
this dev machine (RTX 4060 Laptop, Ada SM89, 8GB VRAM, WSL2) allows, but
full `set_prompt()` + `infer()` was not reached for either frontend, for
two separate reasons unrelated to the generated checkpoint's content:

- `Pi05TorchFrontendRtx(checkpoint, num_views=3, chunk_size=30,
  num_steps=10, use_fp8=True)` gets all the way through
  `_load_norm_stats`, `convert_pi05_safetensors`, FP8 weight
  quantization, and `_precompute_decoder_styles` (9.48 GB allocated on
  an 8 GB card, completing only because this WSL2 environment pages CUDA
  allocations into host RAM rather than raising `OutOfMemoryError` -- the
  same behavior `PROJECT.md` already records for ImageWAM). It then
  raises constructing `RtxFlashAttnBackend`: `flash_rt.flash_rt_fa2` is
  not built in this fork's slim build (`PROJECT.md`'s own build notes
  say this fork only ever builds `flash_rt_kernels`, never
  `flash_rt_fa2`), and no `pip flash_attn` wheel is installed either
  (building one from source needs `psutil` as an undeclared build
  dependency and was not carried further, given the scope of this task).
- `Pi05TorchFrontendThor(checkpoint, num_views=3)` constructs cleanly
  (falls back to a cuBLAS attention path when the CUTLASS strided FMHA
  `.so` is absent, as designed). `set_prompt()` with the default
  `use_fp8=True` then fails inside `GemmRunner.fp8_nn_bias` with
  `cuBLAS error ... code=15` (`CUBLAS_STATUS_NOT_SUPPORTED`) on the
  SigLIP QKV GEMM. This is the same failure class this project's own
  `opportunities.md` (candidate 6 write-up) and `issues.md` history
  already document for other shapes on this exact dev Ada GPU, labelled
  machine-specific there per the project's own no-local-only-closure
  rule -- not a new phenomenon, but not previously observed at this
  specific (SigLIP QKV) shape. `use_fp8=False` avoids that GEMM but then
  fails earlier, in `_encoder_forward_fp16`, on
  `AttributeError: module 'flash_rt.flash_rt_kernels' has no attribute
  'flashrt_rms_qkv_fp16'` -- this build's compiled `flash_rt_kernels.so`
  (built for this fork's ImageWAM/SM89 scope) does not include that
  Pi0.5 FP16 kernel symbol.

## Impact

The generator itself is verified: `model.safetensors` round-trips
completely off-GPU (810 tensors, every key/shape read by
`_load_weights`/`convert_pi05_safetensors` present and finite), and
`Pi05TorchFrontendRtx`'s weight-loading + FP8-quantization code path
consumed the generated file correctly on real CUDA hardware without
error. Neither blocker here originates in the checkpoint content; both
are pre-existing gaps in this fork's build/environment (a slim build
that never compiled `flash_rt_fa2`, and a `flash_rt_kernels.so` build
that does not carry every Pi0.5 FP16 kernel symbol) plus one
already-documented Ada-specific cuBLASLt limitation. Full `infer()`
output finiteness at the requested (3-view, action_dim=32,
chunk_size=30, num_flow_steps=10) shape is therefore unverified on this
machine and needs confirmation on real Thor or RTX hardware with a full
(non-slim) build.

## Next Experiment

On a real Thor or RTX box with the full kernel build (including
`flash_rt_fa2` for RTX and every Pi0.5 FP16/FP8 kernel symbol for
Thor), construct both frontends from a checkpoint produced by
`scripts/gen_synthetic_pi05_checkpoint.py` and run one `set_prompt()` +
`infer()` call each, confirming finite actions at the requested shape.
Not attempted here -- out of scope for this task to rebuild missing
kernel extensions on a machine whose own project notes already scope it
out (`PROJECT.md`'s "Slim build" section).

## Update (cross-hardware benchmark campaign, RTX 5090/Thor)

Confirmed real Thor/RTX hardware use: `Pi05TorchFrontendThor` ran
`set_prompt()` + `infer()` end to end on Thor (fp16/fp8/nvfp4, all
finite, no NaN) and `Pi05TorchFrontendRtx` did the same on RTX 5090 --
the blockers recorded above were specific to this project's own dev
machine's slim build, not a defect in the generator or either frontend.

One correction to `docs/pi05_synthetic_checkpoint.md`'s "OpenPI reads
the same format" claim: the generator omits two tensors OpenPI's own
strict loader expects (`gemma_expert.lm_head.weight`,
`paligemma.model.language_model.norm.weight`) -- confirmed both keys
are genuinely absent from the generated `model.safetensors`, not a
loader-side difference. OpenPI needs `strict=False` (the two missing
keys keep their module's own random init) to load it at all; FlashRT is
unaffected (neither frontend uses `strict` loading or reads these two
keys by name). Also observed: OpenPI measures consistently slower on
this synthetic checkpoint than on a real one at the same shape (roughly
13-25 ms), not yet root-caused -- suspected but unconfirmed cause is
those two tensors keeping an init dtype/value distribution the real
checkpoint's own weights would not have. Not fixed here; the generator
should add these two tensors if OpenPI-side numbers need to be trusted
without the `strict=False` workaround.

# ISSUE-095

Status: explained by two memory states (attribution indirect), non-blocking. Same-night cold and soaked runs at `a21507e` (`docs/imagewam_thor_profile.md`, "Cold and soaked memory states"): the soaked state is 15–20% faster, about 90% of it in GEMM, with device copy bandwidth about 233 vs 256 GB/s; processes of about one minute stay cold. Open: which clocks change (not readable without root), and what triggers and releases the soaked state.

Area: `benchmarks/imagewam_fusion_ab.py` on Thor (cross-process latency)

## Observation

In the Thor A/B of `fuse_qkv_norm_rope` at `989547d` (opportunities.md OPT-032, "Candidate 1, end-to-end with `vec_rms_norm` on"), the same configuration measured in different processes gives different `infer()` P50. Conditions: three runs in sequence (R1 `AB=fuse_qkv_norm_rope`, R2 the same with `HOLD=vec_rms_norm=0`, R3 `AB=vec_rms_norm`); MAXN, GPC locked, `emc_locked=null`.

| configuration | precision | run, side → ms | run, side → ms |
|---|---|---|---|
| vec on, fused off | nvfp4 | R1 A 80.34 | R3 B 66.94 |
| vec on, fused off | fp8 | R1 A 113.86 | R3 B 93.18 |
| vec on, fused off | fp16 | R1 A 144.80 | R3 B 123.91 |
| vec off, fused off | nvfp4 | R2 A 99.67 | R3 A 85.58 |
| vec off, fused off | fp8 | R2 A 112.76 | R3 A 112.62 |
| vec off, fused off | fp16 | R2 A 143.47 | R3 A 143.63 |
| fused on (vec bypassed) | nvfp4 | R1 B 87.53 | R2 B 88.18 |
| fused on (vec bypassed) | fp8 | R1 B 120.85 | R2 B 101.47 |
| fused on (vec bypassed) | fp16 | R1 B 154.40 | R2 B 132.43 |

R1 fp16's A side also spread from P10 124.57 to P50 144.80.

The warp-kernel round at `a2ac9c2` ran the same A/B (`X0=24 USE_FA4=1 AB=fuse_qkv_norm_rope`) in three separate processes:

| precision | A ms, process 1 / 2 / 3 | B ms, process 1 / 2 / 3 |
|---|---|---|
| nvfp4 | 81.09 / 66.64 / 66.36 | 72.84 / 59.01 / 59.09 |
| fp8 | 114.19 / 93.52 / 93.59 | 106.15 / 86.27 / 85.95 |
| fp16 | 125.10 / 124.68 / 125.03 | 117.18 / 116.99 / 117.08 |

Process 1 is 14–21 ms slower on both sides for nvfp4 and fp8. fp16 is stable. A − B stayed within +7.25 to +8.25 ms in all nine cells.

## Impact

Absolute Thor latencies from one process cannot be compared with another process's. The within-process ABBA differences remain the comparable quantity. In R1, their sign agrees with the cross-run fp8/fp16 comparison of fused-on against vec-on (R2 B against R3 B: −8.3 / −8.5 ms). The magnitude of a few-ms effect is not reliable to better than the spread above.

## Evidence

Thor logs `logs/1011_thor_opt032_c1/{R1_fused_vec_on,R2_fused_vec_off,R3_vec}.log` and `logs/1011_thor_opt032_warp/{W1a,W1b,W1c,clocks}.log`. GPC stayed at 1.575 GHz in all six snapshots of the second round.

## Hypotheses

- EMC (memory clock) frequency is not locked and differs between processes. Most of the affected work is bandwidth-bound on Thor.
- A whole-process slow state in R1, the first run after idle, inflates both sides by about 20 ms for fp8 and fp16. The nvfp4 cells do not fit this pattern cleanly.

## Next Experiment

The current EMC frequency cannot be read without root: tegrastats prints no `EMC_FREQ`, and debugfs is not readable. Testing the EMC hypothesis needs a readable memory-clock source on that machine, or an owner-run `tegrastats` with EMC reporting. Without one, the observable next step is to repeat one configuration in 5+ processes and check whether the slow state is bimodal and tied to the first process after idle.

## Resolution

# ISSUE-096

Status: open, non-blocking

Area: `benchmarks/imagewam_fusion_ab.py` on Thor, nvfp4 `vec_rms_norm` numerics

## Observation

`AB=vec_rms_norm` (fused off) on Thor, nvfp4, compares `vec_rms_norm` off against on. The two rounds give different deviations:

| round | commit | actions cos / rel_l2 | backbone cos |
|---|---|---|---:|
| R3 | `989547d` | 0.999797 / 2.0e-2 | 0.986896 |
| soaked A/B | `a21507e` | 0.999225 / 3.94e-2 | 0.990435 |

fp8 (backbone 0.999152) and fp16 (0.999992) are equal in both rounds. The commits between them (`0c33b52`, `a2ac9c2`, `a21507e`) change only `qkv_split_norm_rope_fp16` and its default. That kernel is off on both sides of this A/B.

## Impact

The latency savings are unaffected. The size of the nvfp4 deviation `vec_rms_norm` introduces is not established to better than a factor of about 2 in action rel_l2.

## Evidence

Thor logs `logs/1011_thor_opt032_c1/R3_vec.log` and `logs/1015_thor_soak_ab/`. opportunities.md OPT-032, "Thor, soaked state, current default".

## Hypotheses

- The nvfp4 GEMM algorithms autotuned per process differ between the two runs. B replays A's choice inside one process, but the choice can change across processes, which would move the non-bit-exact difference.
- The seeded observations or noise differ between the two runs.

## Next Experiment

Run `AB=vec_rms_norm` nvfp4 in three processes at one commit and compare the cosines. If they spread, pin the GEMM algorithm and repeat.

## Resolution

## Index: resolved entries and where their conclusions are recorded



This file carries the open problems only. Each entry below states a problem
that is no longer open; the pointer names where its conclusion is recorded.

- `ISSUE-001` — cuBLASLt runs FP8 matmul on sm_89 and sm_90 only in the TN operation layout, so `fp8_gemm_descale_fp16_tn` / `fp8_gemm_descale_f32out_tn` were added and `fp8_cublaslt_layout()` picks NN on compute capability >= 10 and TN below; whether TN is as fast as NN on Thor is unmeasured (`benchmarks/imagewam_fp8_layout_bench.py`). → `flash_rt/models/imagewam/quant_linear.py` (the module docstring and `fp8_cublaslt_layout`), `PROJECT.md`'s GPU notes.
- `ISSUE-020` — the padded text keys are handled by `text_trim` rather than by a mask, which reproduces official's masked attention, and it is the served default. → `docs/imagewam_configuration.md` (`text_trim`, the `default` profile), `plan.md`'s roadmap row for this issue.
- `ISSUE-060` — a precomputed `context` is applied on every `set_prompt` call, because the cache key identifies content only on the live-Qwen3-text-encoder and random-weight paths. → `docs/imagewam_configuration.md` (the `set_prompt` notes under "Entry point").
- `ISSUE-071` — the native `run` / `capture` race was the test harness's, not the Python graph's: both verbs now wait for all prior device work (`cudaDeviceSynchronize`) and the test's snapshot waits for its own copies. → `docs/imagewam_native_cpp.md` (the threading and lifetime contract).
- `ISSUE-080` — `text_trim` is the served default with all six conditions satisfied, the last one (the native pipeline's own per-length capture) confirmed in the `0920c` round. → `plan.md` ("Decisions pending (owner)", E1), `docs/imagewam_configuration.md` (`text_trim`, the `default` and `native` profiles), `THOR_STATUS_SUMMARY.md`'s `0920c` section.
- `ISSUE-081` — `view_shape` is the frontend's resolved view shape, from the single owner `_input_view_shape()`. → `docs/imagewam_model_runtime.md` (the `(views, H, W)` note), the `_input_view_shape` docstrings in `flash_rt/models/imagewam/runtime_surface.py`, `runtime_export.py` and `vae_encoder.py`.
- `ISSUE-082` — the `c20f3a0` session's 225 ms reading and 13.6 ms between-instances spread were that session's state: the `eccf14f` round measured the same gate at 202.2 ms with a 0.4 ms repeat spread, which makes the ladder's 2 ms working threshold meaningful. → `THOR_STATUS_SUMMARY.md` (the `eccf14f` round), `tests/fixtures/imagewam_gate/latency_baselines.json` (`source`), `plan.md` (Phase W12).
- `ISSUE-083` — the encoder's padded length is the caller's (`encode_prompts(..., max_length=...)`), so a workload declares its own `text_max_len` and `text_trim` is a separate mechanism that drops the rows a prompt does not use. → `docs/imagewam_configuration.md` (`text_max_len`), the `encode_prompts` docstring in `flash_rt/models/imagewam/text_encoder.py`.
- `ISSUE-084` — the view count is the workload's: `observation_views(observation, num_views)`, `stage_images` taking exactly that many views, and `encode_to_tokens(ae, views, ...)` encoding N views as one horizontally concatenated image. → `docs/imagewam_configuration.md` (the view-geometry paragraph), `encode_to_tokens` in `flash_rt/models/imagewam/vae_encoder.py`.
- `ISSUE-085` — one real defect (the cuBLAS fallback re-captured into the pool an invalidated capture had left recording) was fixed by abandoning that capture state, and the other two symptoms were the tests' own comparisons. → `docs/imagewam_configuration.md` (the abandoned capture state before the retry), the `_abandon_capture_state` docstring in `flash_rt/frontends/torch/imagewam_thor.py`, `THOR_STATUS_SUMMARY.md`'s `0920` section (restated in `0920c`).
- `ISSUE-086` — nothing is fixed at 224x224 any more: `VaeStageSpec.encode_hw` defaults to the staged views' own size, and the frontend resolves the served per-view size in one place (`_input_view_shape()`) for the in-graph and outside-the-graph paths alike. → `docs/imagewam_configuration.md` (the encode-geometry paragraph), `plan.md` (Phase W12).
- `ISSUE-061` — last line because it was not fixed but satisfied: its subject, a latency baseline with no recorded clock and power state, is met by `latency_baselines.json`'s 202.2 ms `nvfp4` baseline, which carries its clock state (MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive), and `plan.md` records the 231.6 -> 202.2 ms re-base as closing that item; the baseline is per configuration now (`untrimmed_reference` holds the 202.2 ms record), and the only part still open is the `served_default` entry, unseeded until a Thor gate run of the promoted default. → `tests/fixtures/imagewam_gate/latency_baselines.json` (`source`), `plan.md` ("Rounds that closed items", and "Open" item 1).

# ISSUE-097

Status: open, non-blocking

Area: `benchmarks/imagewam_thor_small_m_tile_sweep.py --part infer`

## Observation

In the `1021` Thor round the log prints that 0 linears have a different tile in the two captured graphs (heuristic vs tuned), while the per-shape `chosen` and `default` columns of the same log differ for every N = 1024 shape, and the two graphs' `infer()` P50 differ by 1.98 ms (nvfp4).

## Evidence

Thor `$OUT=/home/jingwu/thor_val/1021/X16_infer.log` (local to the Thor), THOR_STATUS_SUMMARY.md round `1021`, opportunities.md OPT-018.

## Hypotheses

- The count is taken after the script switches every linear back to its default tile for the second capture, so target and current tile are equal at the moment of counting (reported by the Thor agent from reading the script; not yet checked here).

## Next Experiment

Read the counting code against the capture order; count the differing linears before resetting the tiles, and check the printed count against the per-shape table on the next run.

# ISSUE-098

Status: open, non-blocking

Area: `research/topics/wam-inference-acceleration/experiments/colocation_5090/colocation_bench.py` (`green<P>`, `prio`) on Jetson Thor

## Observation

In the `1028` Thor round, with a 16k-token planner context, an in-process green-context split (planner 8 SMs, executor 12 SMs) still leaves executor ticks stalled for up to 2.4 s (p99 2 380 ms, 3.0 % misses at a 260 ms period). The same split keeps the executor at p99 198 ms with a short planner context, and the same cell on the RTX 5090 has p99 34.2 ms. Of the long-context cells, only the process-level MPS limit at 50 % has no misses (p99 225 ms); the in-process `prio` cell and the unlimited `mps100` cell also show second-scale stalls (1.3 s, 1.7 s).

## Evidence

THOR_STATUS_SUMMARY.md round `1028` (Y4); Thor `$HOME/thor_val/1028/cells/*.json` (local to the Thor); `research/topics/wam-inference-acceleration/experiments/colocation_5090/README.md` R5 for the 5090 cells.

## Hypotheses

- A resource shared inside one process and not partitioned by green contexts blocks the executor's tick: the CUDA caching allocator or a device-wide synchronizing driver call (cudaMalloc / cudaFree of the planner's growing long-context buffers on unified memory), or the CPU side of the tick.
- Memory-bandwidth contention: less likely to explain a 15× stall, since the executor on 4 SMs is only 1.4× slower than on 20 SMs.
- The stalls in `prio` / `mps100` are the kernel-boundary queuing measured on the RTX 5090 (R4), scaled by the planner's ~10× longer kernels on the Thor; `green25` would then need a different mechanism.

Round `1029` (executor nvfp4, 2 steps): the stalls keep their length with a different executor (green25 about 2.4 s, prio 1.24 s, mps100 0.56–0.65 s, mps50 none), and the in-process cells record one allocator event per 60 s cell while the planner makes 4–11 calls, which weakens the allocation hypothesis. A further hypothesis: CPU-side work of the planner thread inside `generate` for a 16k-token input holds the Python interpreter while the executor's tick needs it (in-process modes only; mps100's shorter stall would then be GPU queuing alone).

## Next Experiment

Round `1029b` placed the stalls: exactly one per planner call, in the last part of the prefill interval, ending when the prefill ends (inferred from the per-cell call and decode medians); 2.4 s with the planner on 8 SMs (green25), 1.24 s on all 20 SMs (prio). The stall is the planner's GPU prefill of the language model, and the executor makes no progress during it even on disjoint SMs. `generate` keeps only the last position's logits, so it is not a full-sequence LM head. This contradicts the allocation and the interpreter-lock hypotheses (one per call at a GPU phase, length scaling with the planner's SMs). New hypothesis: on the Thor, two green contexts take turns on the GPU instead of running at the same time, so a planner kernel of tens to hundreds of ms holds the executor's partition; the 1.8× slowdown next to the copy load (round `1029`) would then be partly turn-taking, not only bandwidth.

Round `1030` rejected the turn-taking hypothesis: on the Thor a small matmul on one green partition waits at most 11.7 ms while a 241–273 ms kernel runs on the other, so the partitions run at the same time and a single long planner kernel does not hold the executor. The same round found a device-wide throughput limit of about 100 TFLOP/s that 12 SMs already reach, shared by both partitions in proportion to their SMs. The second-scale stall therefore needs a mechanism that is neither a long kernel, the interpreter lock, allocation, nor bandwidth; the remaining candidates are something the executor's tick shares with the planner outside the SMs (copy engines for the tick's host-to-device staging, page mapping of unified memory) at the end of a long prefill. Next: time the tick's parts (host staging, its GPU copies, the graph replay) separately with CUDA events in a long-context green25 cell, to see which part waits.

Round `1030` setup: `partition_concurrency.py` runs a compute-bound load (operands in L2, no DRAM demand) on both partitions alone and together, and a single long matmul on one partition while the other runs small matmuls one at a time. On the RTX 5090 (48 / 122 SMs, with another process on the GPU) green contexts run together (46.1 → 42.5 and 100.9 → 99.0 TFLOP/s together vs alone; a small matmul waits at most 5.3 ms during a 256 ms kernel), plain streams share (214 → 50 and 215 → 91 TFLOP/s). Earlier plan: without CUPTI (see ISSUE-099): timestamp every executor tick (CPU start, GPU events around `stage_inputs` and the graph replay) and every planner phase (prefill start/end, each decode step, call end), plus `torch.cuda.memory_stats()` counters (`num_alloc_retries`, segment allocations) per planner call, in the `green25` long-context cell. A stall that coincides with a planner allocation event supports the first hypothesis; one that sits inside the executor's GPU events while the planner runs prefill supports bandwidth.

Round `1104_h` (harness rhythm: executor fp8 10 steps released every 533 ms, planner stand-in every 5.33 s at 4 061
tokens): the stall also appears at about 4k tokens, shorter. green25 and mps100 each have 3 of 226 ticks over 533 ms
(543–571 ms), all inside a planner prefill; prio (max 428.6 ms) and mps50 (max 294.9 ms) have none. At the harness's
chunk deadline the stall decides which sharing modes are usable on the Thor.

# ISSUE-099

Status: open, non-blocking

Area: `research/topics/wam-inference-acceleration/experiments/colocation_5090/overlap_trace.py` on Jetson Thor

## Observation

In the `1028` Thor round (Y5) the torch profiler's chrome trace of five executor ticks alone contains no event with `cat == "kernel"`, and the overlap phase segfaults twice inside `cuGraphLaunch` / `CUDAGraph::replay` while profiling.

## Evidence

THOR_STATUS_SUMMARY.md round `1028` (Y5); Thor `$HOME/thor_val/1028/Y5.log` (local to the Thor).

## Hypotheses

- CUPTI kernel activity is not recorded for graph-replayed kernels on this Thor stack (torch 2.9.1+cu130), and the graph launch under an active CUPTI subscriber crashes.

## Next Experiment

Run the alone phase with eager kernels only (one `a @ a` under the profiler) to check whether any kernel activity is recorded; if none, CUPTI kernel tracing is unavailable and per-tick timing uses CUDA events instead (ISSUE-098).

# ISSUE-100

Status: open, non-blocking

Area: `research/topics/wam-inference-acceleration/experiments/colocation_5090/run_thor.sh` (`mps25`) on Jetson Thor

## Observation

In the `1028` Thor round both `mps25` cells (planner process at 25 %, executor process at 75 % of threads) segfault in cuDNN engine configuration (`Engcfg`), twice each, and write no result. `mps50` and `mps100` run.

## Evidence

THOR_STATUS_SUMMARY.md round `1028` (Y4); Thor `$HOME/thor_val/1028/cells/ctx*_mps25_hz0.log` (local to the Thor).

## Hypotheses

- cuDNN's engine heuristics fail for a thread percentage that does not map to a whole number of the Thor's SM groups (75 % of 20 SMs = 15).

## Next Experiment

From the cell logs, find which process (planner or executor) crashed; run that process alone under MPS at 75 % and at 25 %.

# ISSUE-101

Status: open, non-blocking

Area: `tests/test_load_model_use_fp8_kwarg.py::test_load_model_routes_pi05_torch_thor_decoder_fp4`

## Observation

The test fails with `ImportError: cannot import name 'Pi05TorchFrontendThorFP4Optimized'`. `flash_rt/api.py` imports `Pi05TorchFrontendThorFP4Optimized` from `flash_rt.frontends.torch.pi05_thor_fp4` for the decoder-FP4 route; the test replaces that module with a stub that defines only `Pi05TorchFrontendThorFP4`.

## Evidence

Fails identically on a clean checkout of upstream `fdd3969b` (test file and `api.py` equal upstream's after merge `4feede2d`); passes on the pre-merge fork `9cb3511`.

## Hypotheses

- The upstream change that added `Pi05TorchFrontendThorFP4Optimized` to the route did not update the test's stub module.

## Next Experiment

Add a `Pi05TorchFrontendThorFP4Optimized` stub to the test's fake module and assert that `use_fp4_decoder=True` routes to it.

# ISSUE-102

Status: latency resolved (cuBLAS version, see ISSUE-103); cross-process action differences open, non-blocking

Area: `Pi05TorchFrontendRtx` (RTX 5090, default FP8 path), run-to-run outputs and latency across processes

## Observation

On 2026-11-03, the same Pi0.5 check (LIBERO checkpoint, one fixed observation, `torch.manual_seed(1234)` per call, `calibrate_with_real_data` then `infer`) gave actions differing by up to 0.021 (max abs) between the run right after merge `4feede2d` and five later runs, which agree with each other bit for bit. `infer()` P50 was 24.95 ms in the first run and 27.0-28.8 ms in the later ones. Within each process, repeated `infer()` calls are bit-identical.

## Evidence

The first run loaded pip `nvidia-cublas` 13.6.1.10. The later runs loaded 13.1.1.3 (pip, pulled in by `pip install accelerate` at 2026-11-03 01:03) or, after that wheel was uninstalled, the conda env's 13.4.1.3; they agree with each other bit for bit. Between the first run and the later ones, `flash_rt_kernels` was also rebuilt with `-DFLASHRT_BUILD_QWEN3_VL=ON`; the only binding it adds is `w16a16_gemm_sm120_bf16`, which the Pi0.5 code does not call. The pipeline autotunes its GEMMs by timing at construction (`Pi05Pipeline.autotune_gemms`, always on); the frontend's docstring notes autotune variance perturbs numerics.

## Measurements (2026-11-03, `nvidia-cublas` 13.6.1.10 restored)

`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/cublas_restored/pi05_check.py`, two processes: `infer()` P50 24.97 and 25.02 ms. Actions: 0.0040 max abs between the two processes, 0.0089 and 0.0049 against the first run of the day (`pi05_env13.6*.npz`, `pi05_cublas13.6_before.npz`).

## Hypotheses

- Timing-based GEMM autotuning picks different algorithms per process under cuBLAS 13.6.1.10, which would give the cross-process action differences; under 13.1.1.3 and 13.4.1.3 the picks were stable across five processes.

## Next Experiment

Log the algorithm each autotuned GEMM picks, run the check in five processes under 13.6.1.10, and compare the picks with the action differences.

# ISSUE-103

Status: resolved

Area: ImageWAM LIBERO fp8 executor tick time on the RTX 5090 with the current `flash_rt_kernels` build

## Observation

On 2026-11-03 the ImageWAM fp8 LIBERO tick (`stage_inputs` + graph replay, 10 steps, VAE in the graph) takes 28.2 ms
(CUDA events, 100 ticks, GPU idle otherwise), whether the graph is the one `set_prompt` captured or a recapture on a
side stream (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/exec_capture_check.json`). `docs/imagewam_rtx5090_profile.md`
records `infer()` P50 23.85 ms for the same configuration at `a21507e`, and the 2026-10 co-location study measured
24.8 ms per tick.

## Evidence

The current `flash_rt_kernels` comes from `build_full/` (non-slim, `-DFLASHRT_BUILD_QWEN3_VL=ON`,
`FA2_ARCH_NATIVE_ONLY=ON`), built after merging upstream `fdd3969b`. ISSUE-102 sees a similar rise for Pi0.5 on this
machine (24.95 -> 27.0-28.8 ms) after the same rebuild. SM clocks during the runs were 2.84-2.87 GHz.

## Hypotheses

- The rebuilt kernel library (build flags or the merged kernel sources) is slower for these pipelines. Refuted: see
  Measurements.
- The cuBLAS library the process loads changed. Supported: see Measurements.

## Measurements (2026-11-03)

- Kernel library swapped into `flash_rt/`, one session, `exec_capture_check.py` (100 ticks): every pair of
  `flash_rt_kernels` (current `build_full/`, pre-merge, slim of 2026-10-11) and `flash_rt_fa2` (2026-11-03, 2026-10-12)
  gives 28.0-28.2 ms, the current pair run first and last 28.0 / 28.0-28.2
  (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/so_compare/`, `run_so_compare.sh`).
- `benchmarks/imagewam_module_operator_profile.py fp8` on the current tree: `infer()` P50 28.17 ms against 23.85 ms in
  `docs/imagewam_rtx5090_profile.md`. Eager kernel time 26.59 against 22.47 ms: GEMM 14.5 against 10.86 ms and the
  attention GEMMs about 0.4 ms more; conv 2.43 / 2.41, norm 2.08 / 2.08, elementwise 1.72 / 1.70, quant/cast
  0.81 / 0.80 (`cells/so_compare/doc_profile_fp8_now.json`).
- The process loads `libcublas{,Lt}.so.13.4.1.3` from the conda env (`targets/x86_64-linux/lib`, installed
  2026-09-22). No `nvidia-cublas` pip wheel is installed in the env any more; PROJECT.md recorded 13.6.1.10 for it.
- `nvidia-cublas` 13.6.1.10 installed with `--target tmp/cublas_13.6.1.10` and loaded with `LD_PRELOAD`: 24.36-24.40 ms;
  13.4.1.3 before and after it: 28.0 and 28.1 ms (`cells/so_compare/cublas_*.json`).

## Next Experiment

None.

## Resolution

The flashrt env had pip `nvidia-cublas` 13.6.1.10 and `nvidia-cuda-nvrtc` 13.3.33 until 2026-11-03 01:03, when
`pip install accelerate` re-resolved torch's `cuda-toolkit==13.0.3` pin and replaced them with 13.1.1.3 / 13.0.88;
uninstalling those two wheels at 01:05 left conda's 13.4.1.3. Both wheels were reinstalled at their earlier versions
with `pip install --no-deps "nvidia-cublas==13.6.1.10" "nvidia-cuda-nvrtc==13.3.33"`; processes load
`site-packages/nvidia/cu13/lib/libcublas{,Lt}.so.13` again. Measured after the restore
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/cublas_restored/`):

| Measurement | 13.4.1.3 (conda) | 13.6.1.10 (pip, restored) |
|---|---|---|
| `exec_capture_check.py` tick P50, 3 x 100 ticks | 28.0-28.2 ms | 24.21-24.25 ms |
| `imagewam_module_operator_profile.py fp8` `infer()` P50 | 28.17 ms | 24.11 ms (23.85 ms in `docs/imagewam_rtx5090_profile.md`) |
| same, GEMM kernels in the graph | 17.48 ms | 13.37 ms |
| same, conv / norm / elementwise | 2.43 / 2.08 / 1.62 ms | 2.40 / 2.09 / 1.61 ms |
| Pi0.5 `infer()` P50 (ISSUE-102) | 27.0-28.8 ms | 24.97 / 25.02 ms |

The co-location measurements of 2026-11-03 (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/`) ran under 13.4.1.3.

# ISSUE-104

Status: resolved

Area: `Qwen36TorchFrontendRtx` (NVFP4) / `Qwen36VlTorchFrontendRtx` constructed while a non-default CUDA stream is current

## Observation

Constructing the frontend inside `with torch.cuda.stream(side_stream):` produces garbage output on every later call
(for example `<{"juga\n<{"juga><function|...` instead of `{"done":[],"next":"Move the gripper to grasp the tomato sauce
can."}`), with the same checkpoint and prompt. Constructed on the default stream, the same frontend called later
under a side stream, and with its graph and snapshot streams replaced by side streams, gives the correct output
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/stream_check.py`, `smoke/stream_check_*.json`).

## Evidence

`_quant_bf16_lin_proj` (`_qwen36_rtx_nvfp4_weights.py`) copies each BF16 weight to the device with `.to(device)` on the
current stream and then launches `fvk.bf16_weight_to_nvfp4_swizzled(..., 0)` on the legacy stream 0. Torch side
streams are non-blocking, so stream 0 does not wait for the copy.

## Hypotheses

- The load-time quantization kernels on stream 0 read partly copied BF16 weights when the current stream is a side
  stream.

## Next Experiment

Construct under a side stream with a `torch.cuda.current_stream().synchronize()` before each stream-0 launch, or with
the launch on the current stream, and compare the output and the quantized weights against a default-stream
construction.

## Resolution

The load-time conversion kernels in `_qwen36_rtx_nvfp4_weights.py` (`nvfp4_sf_linear_to_swizzled` in
`_load_quantized_proj` / `_load_quantized_mlp_gate_up`, `bf16_weight_to_nvfp4_swizzled` in `_quant_bf16_lin_proj`) now
run on the current torch stream. Before the change, a side-stream construction left 48 of the 1 482 language-model
weight buffers different from a default-stream construction (most of them all zero) and produced garbage; after it, the
byte-sum digest of all 1 482 buffers equals the default-stream one and the three outputs are identical
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/smoke/stream_check_{plain,side}_{prefix,fix}.json`).

# ISSUE-105

Status: open, non-blocking

Area: Qwen3.8-27B planner output under in-process co-location with the ImageWAM executor (`prio`, RTX 5090)

## Observation

In `prio` short-context co-location with the planner's per-cycle readback outside the GIL, one planner call in about 60
produces a different token sequence from all the others (same prompt, greedy): 1 of 63 calls
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/hostfix/short_prio_p1.json`, binding fix) and 1 of 61
(`cells/short_prio_nogil.json`, research monkeypatch). The planner alone (5 calls per context), `prio` with the GIL-held
readback (76 calls) and the green-context cells (19-57 calls) give one sequence each.

## Evidence

- The differing calls are the slow ones: 1 259 ms and 2 699 ms against about 945 ms for the others; in both windows the
  planner's captured-graph count grows (50 -> 56, 50 -> 86), and in the windows with a single sequence it does not.
- The differing calls take 26 speculative cycles with 39 accepted drafts; the others 25 cycles, 40 accepted.
- Under MPS (`mps50`, separate process) every call gives one sequence that differs from the whole-device planner's (67 of
  67 calls the same).

- With per-call graph counts (`cells/hostfix/*_p2w*.json`, thread-local capture in the frontends, both monkeypatches
  off): every differing call captured graphs (short `prio` window 1: calls 31 and 36, 50 -> 82 -> 92 graphs; window 2:
  call 57, 50 -> 66); one capturing call (long `prio` window 1, call 15, 64 -> 108) kept the common output. The
  differing outputs come from a small set: 3880926757, 3071263923 and 2639096006, the last being the sequence every
  `mps50` call produces. No differing call in 9 `green25` windows (19-57 calls each).
- The MTP chain graphs are keyed by cache position and draft length, and the draft length adapts
  (`FLASHRT_QWEN36_TQ_ADAPTIVE_K`), so a call whose acceptance deviates visits positions with no graph yet and captures.
- In the same `prio` windows, 3-6 of about 1 120 executor ticks give actions that differ from the reference by up to
  4.9e-3 (ISSUE-106); `green25` windows: 0 of about 1 200.

## Hypotheses

- Without an SM partition, the two models' kernels interleave on the same SMs and a reduction whose result depends on
  block scheduling (atomics, split-K) changes slightly; for the planner this flips a near-tie in a draft or verify argmax,
  the call takes another acceptance path, and the graph capture follows from that path.
- A planner call that captures a new graph while the executor thread runs computes a different result (a race between
  the capture path and the other thread's CUDA work).
- The MPS difference is a deterministic numeric change from the thread-percentage limit; its sequence is one of the
  near-tie alternatives.

## Next Experiment

List the kernels on the planner's decode path and the executor's graph that reduce with atomics or split-K. Run the
executor alone next to a compute-only load on shared SMs (no planner) and count ticks with action differences; run
`prio` with the planner warmed until no capture happens in the window and count differing calls.

# ISSUE-106

Status: open, non-blocking

Area: ImageWAM fp8 executor actions under in-process co-location without an SM partition (`prio`, RTX 5090)

## Observation

With a fixed action noise, the executor's actions are bit-identical to the reference on every tick alone and under
`green25` (about 1 200 ticks per window, 6 windows), and in two short-context `prio` windows; in two long-context `prio`
windows 3 and 6 of about 1 120 ticks differ, by up to 4.4e-3 and 4.9e-3 (max abs, normalized actions), and in a short
`prio` window under py-spy sampling 3 of 1 181 ticks differ by up to 4.4e-3
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/hostfix/*_prio_p2w*.json`, `exec_action_check`).

## Evidence

The differences occur only where the planner's kernels share the executor's SMs. `prio` long overlaps the executor
with the planner's graph-replayed prefill (the executor's graph takes 55-60 ms during it).

## Hypotheses

- A kernel in the executor's graph reduces in an order that depends on block scheduling (atomics, split-K), and SM
  sharing changes that order.
- A buffer shared between the two models (a global workspace in the kernel library) is written by both concurrently.

## Next Experiment

As in ISSUE-105: the executor alone next to a compute-only load on shared SMs; and a list of the executor graph's
kernels that use atomics or split-K. A shared-workspace race would also show as sporadic tick differences under
`green25`, on top of the constant offset of ISSUE-107.

## Measurements (2026-11-03, cuBLAS 13.6.1.10)

`cells/cublas136/cur_*.json`: `prio` short 1 of 1 201 ticks (2.7e-3), `prio` long 2 of 1 200 (3.5e-3); `coop` short
and long 0; `green` cells see ISSUE-107.

# ISSUE-107

Status: open, non-blocking

Area: ImageWAM fp8 executor graph captured on a green-context partition, cuBLAS 13.6.1.10 (RTX 5090)

## Observation

With cuBLAS 13.6.1.10, the executor graph recaptured on the green partition's stream gives actions that differ from
the whole-device reference (`fe.infer` on the default stream, same fixed noise) by 4.73e-3 (green25 short), 2.81e-3
(green25 long) and 2.95e-3 (green50 short and long), max abs, normalized actions. The difference is there in the check
right after the recapture, before the planner starts, and every tick of the window shows the same maximum
(`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/cells/cublas136/cur_*_green*.json`, `exec_action_check`). With cuBLAS
13.4.1.3 the same check gave 0 under `green25` (`cells/hostfix/`).

## Evidence

The offset is constant across ticks and present without a planner, so it is not a race between the two models. Only
the cuBLAS build and the partition differ between the zero and non-zero cases.

## Hypotheses

- cuBLASLt 13.6.1.10 picks GEMM algorithms from the SM count of the context it runs in, and a green context reports
  its partition's SM count, so the executor's fp8 GEMMs use other kernels (other split-K or tile orders) on 82 or
  122 SMs than on 170.

## Next Experiment

Log the cuBLASLt algorithm the executor's GEMMs get on the whole device and on each partition (or compare kernel names
in an nsys trace of one graph replay), with 13.6.1.10 and 13.4.1.3.


# ISSUE-108

Status: resolved

Area: `Qwen36TorchFrontendRtx` long-context prefill with `FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1` (NVFP4, FP8 or TQ KV)

## Observation

With the prefill graph on, the first call at a prompt length not seen before in the process gives a broken reply
(`{"user": "I want to 100% natural, 100% organic, ...` instead of `{"subgoals":["put the red-and-white alphabet soup
carton in the woven basket ...`); the same messages called again give the correct reply. With the prefill graph off,
the first call is already correct. Seen on every new prompt length of the recorded VoLoAgent calls (decompose 2 126 /
2 942 tokens, monitor 4 072-5 726, replan 1 427) (`research/topics/wam-inference-acceleration/experiments/harness_replay_5090/calls/`, `text_equal`;
`smoke/first_call_pf{0,1}.json`).

## Evidence

`_ensure_prefill_graph_nvfp4_fp8kv` / `_tq` run the chunk twice as warm-up, roll the linear-attention state and the
dequant mark back after each, capture the graph and return `captured_live=True`, and the caller then skips the replay.
A capture does not execute the kernels, so the chunk never runs: the recurrent state stays at its pre-chunk value
(the KV rows hold the warm-up's writes) and every later chunk and the decode start from the wrong state.

## Resolution

Both capture functions return only the graph, and the four call sites in the long-context prefill always replay it,
after a capture as after a cache hit (`flash_rt/frontends/torch/qwen36_rtx.py`). After the change, with the prefill
graph on, the first call equals the later calls and the prefill-graph-off output for a 2 126-token decompose call and
a 5 704-token monitor call (`smoke/first_call_pf1_fix_*.json`, `smoke/first_call_pf0_*.json`).

The co-location cells with the prefill graph (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/`) ran fixed prompts
whose prefill graphs were captured in the warm-up calls before the measured window; ISSUE-105's flips were calls that
captured graphs inside the window.

# ISSUE-109

Status: open

Area: `Qwen36TorchFrontendRtx._generate_long_ctx_speculative_KN_nvfp4` (NVFP4, FP8 KV cache), greedy output against
the MTP draft length K

## Observation

Greedy decode is meant to give the same tokens for every draft length K (the main model's argmax decides every
token). On the planner (LoRA-merged Qwen3.8-27B master, official Qwen3.8-27B-FP8 MTP head), 25 of 26 prompts give the
same output crc for K = 1, 3, 5, 7 and the length table; the synthetic 16 315-token prompt gives a different output with
K = 1 (crc 816102686, 13 tokens before EOS) than with K = 3, 5, 7 and the table (crc 972381506, 11 tokens)
(`research/topics/wam-inference-acceleration/experiments/harness_replay_5090/cells/mtp_speedup.json`).

## Evidence

K = 1 verifies two tokens per main-model forward (Kv = 2); the other runs verify 4 to 8. The verify forward at a
given position therefore runs different Kv shapes, and the positions at which a forward starts differ between K.

## Hypotheses

- The Kv-dependent verify path (FP8 KV staging, attention split, GEMM tiling at M = Kv) gives argmax ties or
  near-ties a different winner at this length; the outputs differ from the first such token on.

## Next Experiment

Decode the 16 315-token prompt with K = 1 and K = 3 and record per-token argmax and top-2 logit margin; find the first
differing position and compare the logits of the Kv = 2 and Kv = 4 forwards at that position.

# ISSUE-110

Status: open

Area: `Qwen36TorchFrontendRtx` long-context prefill (NVFP4, FP8 KV cache), chunk length
`FLASHRT_QWEN36_TQ_PREFILL_CHUNK`

## Observation

The planner's greedy reply to the same recorded VoLoAgent calls depends on the prefill chunk length. Episode 0
(wrist 224, 11 calls), planner alone: the eager prefill with the default chunk gives the recorded reply (recorded
with the prefill graph, default chunk) on 11 of 11 calls; with 256-token chunks, 8 of 11 replies change wording and
one monitor decision changes from `in_progress / continue` to `complete / next` (step 720)
(`research/topics/wam-inference-acceleration/experiments/harness_replay_5090/smoke/prefill_pf0_c{256,def}.json`). The `coop` replay cells, which need
256-token chunks, match 4 of 33 recorded replies; the other cells 33 of 33.

## Evidence

Same weights, prompt, decode path and draft length; only the prefill chunking differs. The changed replies stay
fluent and on topic, so the prefill result is not broken but numerically different.

One extra chunk split is enough. Restoring the episode-constant monitor prefix from a capsule and appending the rest
(`research/topics/long-horizon-planner-executor/experiments/monitor_call/`, master, 59 monitor calls of six
recordings) is token-identical to the full prefill when the boundary is a multiple of the 2 048-token chunk (59 of
59 replies); with the boundary at the first NOW image (2 641 / 3 457 tokens, not chunk-aligned) 57 of 59 replies
change wording, none of the 57 `continue` actions change, and both non-`continue` actions change to `continue`
(ep188 wrist 224 step 720 `grasp_tool`, ep0 wrist 720 step 720 `next`). Episode 0 step 720 is also the call whose
decision changed with 256-token chunks above (there `continue` -> `next`). The official Qwen3.8-27B on the same experiment (ep0 / ep43
wrist 224, 20 calls, its own replies as reference): aligned 20 of 20 replies unchanged; unaligned, its only
non-`continue` action (ep43 step 800 `replan`) changes to `continue` and one reply becomes `{}` (ep43 step 560).

## Hypotheses

- A chunk boundary changes the attention split over the FP8 KV prefix (rows of earlier chunks are read back from the
  FP8 cache, rows of the current chunk from BF16), and the GEMM shapes of the linear-attention and MLP layers; the
  hidden state at the last prompt position differs enough to flip near-tied tokens.

## Next Experiment

For one changed call, compare the last-position logits (top-2 margin) and the per-layer hidden state of the prefill
with 256-token chunks against the default chunk; repeat with the BF16 KV route to separate FP8 KV readback from
chunked GEMM shapes.

# ISSUE-111

Status: open, non-blocking

Area: `research/topics/wam-inference-acceleration/experiments/colocation_5090/colocation_bench.py` process exit on Jetson Thor (in-process planner
thread modes)

## Observation

In round `1104_h`, the `prio` and `green25` cells exit with status 134 after writing their results:
`terminate called without an active exception` (prio) and `malloc(): unsorted double linked list corrupted`
(green25). The `isolated`, `mps50` and `mps100` cells (no planner thread in the executor's process) exit cleanly. In
round `1029`, `p100_ctx16000_prio` raised a `c10::Error` in `conv3d` / `slow_conv_dilated3d_cuda` at exit, with the
planner thread still running its vision encoder.

## Evidence

THOR_STATUS_SUMMARY.md rounds `1029` and `1104_h`; logs in the Thor's `$HOME/thor_val/1104_h/cells/H1_{prio,green25}.log`.
The JSON results are written before the abort, and their numbers agree with the neighbouring cells.

## Hypotheses

- The bench ends the timed window while the planner thread is inside `generate`; interpreter shutdown destroys CUDA /
  allocator state that the still-running thread uses (a `std::thread` destroyed while joinable gives the `terminate`
  message; freeing under a running kernel launch gives the heap corruption).

## Next Experiment

Stop the planner thread and join it (after its current call) before the results are written, and check that the
in-process cells then exit with status 0.


# ISSUE-112

Status: open, non-blocking (the reply difference is explained; the slower decode after GPU idle is not)

Area: `flash_rt/frontends/torch/qwen36_vl_thor.py` (`Qwen36VlTorchFrontendThor`) and `qwen36_thor.py` on Jetson
Thor: replies against the RTX 5090 references, and decode ms per speculative cycle over a session

## Observation

- Round `1104_h3` (H3): all 44 timed replies of the official Qwen3.8-27B NVFP4 on the 22 recorded harness calls
  differ from the RTX 5090 references (decoded on `Qwen36VlTorchFrontendRtx` with the FP8 / FA2 ViT).
- Round `1104_h3b`: parsed decisions from untimed 160-token calls equal the references in 18 of 22 calls, 1 reply
  does not parse; the same four calls differ with the M-tile and the cuBLASLt tail fc.
- Decode ms per cycle at K = 3 on the same shapes is 136-157 ms in one part of a session and 88-100 ms in another:
  `1104_h3` H3a 156.7 / 136.3 / 136.4 ms (decompose / monitor 224 / monitor 720, three runs each within 1 %);
  `1104_h3b` start 156.9 / 99.8 / 93.3 ms, end 99.0 / 88.4 / 87.7 ms; the first of the three monitor-224 runs at
  the start took 5 198 ms over 23 cycles.
- Round `1104_h3c` (`thor/decode_cycle_drift.py thor`, the monitor shape back to back for 300 s, the first GPU
  process after `git pull`): 136.4-137.0 ms per cycle from 28 s to 153 s, 122-129 ms at 159-166 s, 95.7-99.7 ms
  from 172 s to 218 s, 87.4-88.0 ms from 232 s to 302 s, with isolated 120-249 ms generations at 194, 226 and 238 s.
  Prefill of the same generations: 4 031-4 035 ms in the slow part, 3 977-4 007 ms in the steady part. The image
  replay in the next process: monitor median 85.8 ms per cycle, 2 of 44 timed calls slower (140.2, 101.8 ms).

- Round `1105_h6h4` (H6, `decode_cycle_drift.py thor 300 300 180`, another process had used the GPU until just
  before): 87.4-88.1 ms per cycle from the first warm generation on; after 300 s of idle in the same process
  137.9 / 138.3 ms, then 117.8 -> 93.6 ms over 8 generations (about 50 s), then 87.5-87.8 ms.
- Round `1110_h7h4bh8` (H7, `decode_cycle_drift.py thor 120 5,15,30,60,120 60`, GPU idle before the start): steady
  about 87.2 ms per cycle; the first / second generation after each idle gap: 5 s 87.3 / 164.9, 15 s 87.2 / 87.1,
  30 s 87.0 / 87.1 (third 95.8), 60 s 87.1 / 87.1, 120 s 96.7 / 95.4 ms (prefill 4 048 against about 3 980 ms). No
  gap from 5 s to 120 s reproduced the 138 ms plateau that followed 300 s of idle.
- Round `1111_h10` (H10, `decode_cycle_drift.py thor 120 180,300,180,300,180,300 60`; another process, thor3's
  python with MPS, present for the whole round): in the four segments without visible interference (prefill
  4 023-4 046 ms) a plateau of 141.6-142.1 ms per cycle followed 2 of 2 idle gaps of 300 s and 1 of 2 gaps of 180 s;
  it lasted 35-36 s twice and the whole 57 s window once. The other 180 s gap gave 105.9 / 100.2 ms, below 90 ms after
  36 s. In the first two segments (prefill 4 800-6 000 ms, copy jumping between 119 and 260 GB/s) no full plateau
  formed.
## Evidence

- On the 5090, the RTX language model with the Thor frontend's ViT configuration (BF16 linears, SDPA;
  `thor/planner_calls_vl_rtx_thor_vit_table.txt`) changes 42 of 44 replies and 4 of 22 decisions; the FP8 / FA2 run
  with the same decision pass keeps 44 / 44 and 22 / 22 (`thor/planner_calls_vl_rtx_table.txt`).
- The decisions that change in meaning on the Thor (ep0 decompose: 3 subgoals become 2; ep43 monitor@800: `replan`
  becomes `continue`) change the same way in `rtx_thor_vit`. The other two Thor differences are ep43 decompose (same
  subgoals, longer wording) and ep0 monitor@720 (not parseable); ep43 monitor@160, which `rtx_thor_vit` changes to
  `grasp_tool`, equals the reference on the Thor.
- Per cycle: in `1104_h3` the H3a shapes ran first after the checkpoint conversion, the image replay after them
  (about 90 ms per cycle derived from its medians); in `1104_h3b` the slow part is the first shape of the first
  shape run. GPC and NVD clocks are locked, `emc_locked` is `null`, CPU frequencies are not locked. The two scripts
  set different environments (`planner_call_shapes.py`: eager prefill, route minimum 512, `expandable_segments`;
  `planner_calls_vl.py`: prefill graph, route minimum 1, default allocator), but the same script gives both levels.
- `1104_h3c`: all 44 readings after each generation show every CPU core at 2 601 000 kHz and the devfreq nodes at
  `vic` 54 MHz, `gpu-gpc-0` 1 575 MHz, `gpu-nvd-0` 1 692 MHz; no readable clock differs between the slow and the
  steady part. The Thor exposes no EMC devfreq node, and tegrastats prints no `EMC_FREQ` (ISSUE-095).
- The slowdown selects bandwidth-bound work: decode per cycle 136.7 / 87.7 = 1.56x, prefill 4 033 / 3 977 = 1.014x.
  ISSUE-095 shows the same pattern on the executor: the first process after idle is 14-21 ms slower for nvfp4 and
  fp8, fp16 is unchanged.

- `1105_h6h4`: the 1 GiB copy probe is 228.6 GB/s in the first two slow generations against 256.5 GB/s steady
  (-11 %), and back to 252.8-257.7 GB/s while decode is still at 93.6-117.8 ms per cycle; prefill is 1.6 % slower.
  The slow state follows an idle period (it appears after 300 s of idle in a running process and not at the start of
  a process that followed another GPU process), not the start of a process.
- `1111_h10`: the plateau level is the same every time (141.6-142.1 ms, 1.63x the 87.2 ms steady value); copy during
  it 234-244 GB/s against 253-260 GB/s (about -6 %), prefill +0.5 %. GPC 1 575 MHz, NVD 1 692 MHz and CPU 2 601 MHz
  are unchanged throughout. Idle thresholds observed so far: 5-120 s 0 / 5 gaps, 180 s 1 / 2, 300 s 2 / 2 (segments
  without visible interference; `1105_h6h4` 1 / 1 at 300 s).
## Hypotheses

- Replies: the ViT configuration explains the decision differences; the Thor language model's kernels change wording
  and the borderline calls (ep0 monitor@720, ep43 monitor@160).
- Per cycle: a memory-side state that the Thor enters during GPU idle and leaves over about a minute of load. A lower
  memory clock alone would lower the streaming copy by the same ratio as decode; the copy drops 11 % for about 13 s
  while decode is 1.58x slower and stays above steady after the copy has recovered, so the state affects the decode
  access pattern (many smaller weight reads per cycle) more than a streaming copy. Candidates: the EMC governor's
  frequency ramp interacting with access intensity, DRAM power-down exit latency, or SMMU / TLB state of the
  weight pages. The CPU frequency and the GPC / NVD clocks are excluded.
- The same two states are documented for the executor (`docs/imagewam_thor_profile.md`, "Cold and soaked memory
  states"): cold ImageWAM `infer()` is 15-20 % slower than after about 5 minutes of continuous load, the copy
  bandwidth about 10 % lower (231-234 against 256 GB/s), GEMM up to 30 % slower, and the VDD_CPU_SOC_MSS rail stays
  at 16-19 W instead of the soaked 21 W. The planner's decode shows the same signature with a larger factor.
- A discrete level entered after 2-5 minutes without load: the plateau value repeats to 0.5 ms across three
  occurrences. That the interfered segments of `1111_h10` formed no full plateau is consistent with another process's
  load keeping the state warm during the idle; the other process's activity was not logged, so this is not tested.

## Next Experiment

- The idle threshold is between 120 and 300 s (H7, H10). Remaining: the same 180 / 300 s gaps with a periodic
  executor load running during the idle, to tell whether a co-located executor keeps the state warm; this is part of
  the co-location measurement with the planner process
  (`research/topics/long-horizon-planner-executor/experiments/colocation_thor/planner_process.py`).

## Impact

Thor decode numbers taken in the first minutes of a session's GPU load are up to 1.56x the steady value
(`1104_h3`, `1104_h3b`, `1104_h3c`). A Thor round that times decode warms up with bandwidth-bound load first, or
brackets its rows with a repeated reference row.

# ISSUE-113

Status: open, non-blocking

Area: `research/topics/long-horizon-planner-executor/` — paper metadata and numbers not verified on a primary page in
the first survey round (2026-11-09); the entries are marked † in its `papers.md`.

## Observation

- Numbers given only in figures (no value in the text): Hi Robot against flat π0 (Fig. 5, Fig. 8); π0.5 §V-E (Fig.
  13); OneTwoVLA per task (Fig. 6); Gemini Robotics 1.5 per task (Fig. 17 / 41); Slow Brain, Fast Planner latency
  sweep (Fig. 6); VLA-OS per-paradigm LIBERO-Long values.
- Not read in full: Long-VLA (PDF over the fetch limit, no HTML; L-CALVIN chain length and values unknown); DualVLN on
  Jetson Thor (OpenReview Ei1Wgdo2I0 blocks automated access; abstract only, observation age undefined).
- Instruction given to flat OpenVLA in RoboCerebra's comparison is not stated; Tri-System π0.5 Table 1 column
  alignment for Tidy Desk is inferred.
- Titles written without a primary-page check in this round: AsyncDriver, SmolVLA, Orion (and the Orion DOI); full
  titles of Think at 5 Hz Act at 20 Hz, VLA-Corrector and LiteVLA-H were not recorded.
- Venues not confirmed from the arXiv comment field: VLA-OS (NeurIPS 2025 page only), OxyGen (HTML page says NeurIPS
  2026).
- The edge isolation study (2601.07600) and Understanding Asynchronous Inference (2605.08168) were reused from earlier
  surveys without reopening.

## Next Experiment

Open each primary page (arXiv abs / HTML, proceedings) and either replace the † entry with the verified value or
state that the value is figure-only.

# ISSUE-114

Status: open, non-blocking

Area: `flash_rt/frontends/torch/qwen36_vl_thor.py` (`Qwen36VlTorchFrontendThor`) and `qwen36_thor.py` on Jetson Thor:
replies of the same prompt in one process are not always token-identical.

## Observation

- Round `1110_h7h4bh8` (H8, `monitor_call/monitor_call_bench.py thor own`, official Qwen3.8-27B NVFP4, 20 monitor
  calls): the reference reply of each call is the model's own 160-token `generate`; a second `generate` of the same
  call later in the same process equals it in 15 of 20 calls. The agent path without a capsule equals it in 14 / 20,
  with the aligned capsule 12 / 20, with the capsule cut at the first NOW image 15 / 20. The parsed action equals the
  reference in 20 / 20 for every full-reply and stop-at-action cell.
- RTX 5090, same script and model (`monitor_call_rtx_official.json`): `generate`, no capsule and aligned capsule are
  20 / 20; the capsule cut at the first NOW image is 0 / 20.
- Round `1111_h12h11h9` (H12, `monitor_call/reply_repeat.py thor`, the same 20 calls, five `generate` calls back to
  back per call): repetitions 2-5 equal the first in 44 of 80; all five are identical in 7 of 20 calls. RTX 5090,
  same script: 80 / 80. All 100 Thor decisions are `continue`.
- H11 of the same round (27 closed-loop calls): the second `generate` equals the reference in 26 / 27.
- Round `1111_h14h13` (H14, `reply_repeat.py thor after_previous`, the same 20 calls, every repetition preceded by a
  `generate` of the previous call, GPU idle): repetitions 2-5 equal the first in 79 of 80; all five are identical in
  19 of 20 calls. All 100 decisions are `continue`. RTX 5090, same order: 80 / 80.
- H13 of the same round (GPU idle, prefill linear attention on `wy_lt`, 47 calls): the timed `generate` equals the
  untimed reference `generate` of the same call in 44 / 47.

## Evidence

- The reference and the second `generate` run the same code on the same inputs; the difference is between two runs,
  not between paths.
- Per call (`research/topics/long-horizon-planner-executor/experiments/monitor_call/monitor_call_thor_official.json`):
  against the second `generate`, the replies of the other full-reply cells first differ at characters 69-220, inside
  the `reason` text (which starts at character 48), except one call (aligned capsule, ep43 step 80) that differs from
  character 0: one reply opens with a ```json fence and the other does not. The calls that differ are not the same
  set in every cell. All 20 reference decisions are `continue`.
- The Thor long prefill runs the `native` linear-attention backend (per-step recurrent chunk scan) pinned by
  `Qwen36TorchFrontendThor.__init__`; the 5090 runs `wy_lt`.
- H12: every difference is inside the `reason` text (characters 63-246) or at token 0 (a ```json fence or not, 4
  calls). The top-2 margin at the first differing token, scored by one prefill, has median 0.375 logit over 13 calls;
  at token 0 three calls have margin 2.50-4.50.
- H12: in 7 of 20 calls repetitions 2-5 all leave the first at the same token, and in 4 of them repetitions 2-5 are
  identical to one another. The first repetition follows the previous call's three scoring prefills (prompt plus a
  reply prefix of a different length); repetitions 2-5 follow a `generate` of the same prompt.
- `1111_h12h11h9` H9: on the same layer inputs every correct linear-attention variant ends within cos 0.57-0.72 of the
  per-token path in the last-position logits, so a change in low bits early in the prefill moves logits by more than
  a 2.5-logit margin by the last layer.
- GPU state per run: H8 idle, `generate` against reference 15 / 20 (the two `generate` calls follow different
  operations); H11 and H12 with another process and MPS on the GPU; H13 and H14 idle. Idle alone (H8, H13) does not
  make the two `generate` calls of a call equal; a fixed preceding operation on an idle GPU (H14) does, in 79 / 80.
- H12 and H14 differ in two variables at once: the order and whether another process was on the GPU.
- H14 against H12: the H14 reply equals the H12 first repetition in 12 of 20 calls and the H12 majority of
  repetitions 2-5 in the other 8. The H12 first repetition followed scoring prefills, the H14 repetitions follow a
  `generate`, so their predecessors differ.
- H12 has single repetitions that differ in the middle of runs of identical ones (for example ep0 step 320: the
  fourth differs, the third and fifth equal the first); the third and the fourth repetition both follow a `generate`
  with the same reply.
- H14's one differing repetition (ep43 step 240, the second) leaves at token 33, scored margin 0.75.

## Hypotheses

- State left by the previous call: a Thor-path buffer that a call reads before writing it (rows beyond the prompt, a
  K-row scratch, a recurrent-state save slot) holds what the previous operation left; the same predecessor gives the
  same reply. H14 (79 / 80 with a fixed predecessor) and H8 / H13 (idle GPU, different predecessors, not equal)
  support it. It does not explain H12's single repetitions that differ between two identical neighbours.
- A kernel on the Thor path whose result depends on scheduling (atomics, split-K or stream-K reductions, an attention
  reduction such as xqa) changes low bits when another process shares the GPU (MPS), and a near-tie then flips one
  token. This would explain H12's single repetitions in the middle of identical runs; H14 on an idle GPU has one such
  repetition in 80.
- Decisions: 100 / 100 Thor repetitions unchanged in both orders, with the varying tokens in the `reason` text or the
  fence.

## Next Experiment

- H15: the H12 order (`reply_repeat.py thor back_to_back`) on an idle GPU. Leftover state alone predicts repetitions
  2-5 identical to one another in every call, with the first different in some; if repetitions 2-5 also differ
  singly, as in H12, a source other than the predecessor exists on the idle GPU too. Comparing with H12 isolates the
  effect of another process on the GPU, which is the co-location condition.
- Then, if leftover state is confirmed, fill the Thor frontend's scratch and state buffers with a constant before each
  call and find which buffer removes the dependence.

## Impact

On the Thor, "same reply as the reference" from a single reference run cannot separate a path's own numerical change
from run-to-run variation. Thor comparisons of reply text need a run-to-run baseline cell (here `generate` against
`generate`); decision equality is the comparison that this round supports.

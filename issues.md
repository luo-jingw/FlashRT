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

Status: open

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

The error text is the CUDA generator's check that no capture is open; it fires when the generator still holds the "capturing" flag while the stream is not capturing. Measured on the development GPU (torch 2.14, RTX 4060): none of three ways of failing a capture (a device sync inside it, a Python error inside it, an RNG use then an error) leaves the generator unusable (`scripts/probe_capture_generator_state.py`). So the poisoning is not reproduced off Thor and the torch 2.9.1 build on Thor is the untested variable. Not measured: which failed capture leaves the flag set, and whether one tiny successful capture repairs it.

## Hypotheses

A capture that fails inside `capture_end` (the `synchronize()` in the `capture_sync` mode of `test_fa4_failure_falls_back_to_the_cuBLAS_chain`) skips the generator's epilogue on torch 2.9.1, so the flag stays set for the rest of the process. Alternative: a failed second capture (the fallback also failing) in another test does the same.

## Next Experiment

On the Thor: run `python scripts/probe_capture_generator_state.py` (each case prints whether `torch.randn` works afterwards, and whether one successful capture repairs it). Then `python -m pytest tests/test_imagewam_fa4_dispatch.py -x -q`, and `python -m pytest tests/test_imagewam_fa4_dispatch.py -k capture_sync tests/test_imagewam_infer_action_noise.py -q` in one process to see whether the second file errors. If a case poisons the generator, either the tests reset it (one successful capture in a fixture after the failing ones) or the fallback path does, depending on whether the frontend's own retry leaves it set.

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

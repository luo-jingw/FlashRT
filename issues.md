# ISSUE-001

Status: resolved

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

The hypothesis held: the NN operation layout was the only cause.

- `csrc/kernels/decoder_fused.cu` gained `fp8_gemm_descale_fp16_tn` and
  `fp8_gemm_descale_f32out_tn`: weight stored `(N,K)` row-major,
  `TRANSA=T`, their own descriptor cache keyed by `(M,N,K,output type)`.
  The NN functions and their cache are unchanged.
- `quant_linear.fp8_cublaslt_layout()` returns `"nn"` on compute
  capability >= 10 (Thor keeps the exact NN path it was measured with)
  and `"tn"` below. `Fp8Linear` and `StaticFp8Linear(use_cutlass=False)`
  store the weight in that layout; `layout=` forces one for an A/B.
- H100 (sm_90): `fp8_gemm_descale_fp16_tn` is bit-exact to
  `torch._scaled_mm` at `[4,16,16]`, `[905,27648,3072]` and
  `[64,5120,1024]`; the NN call still returns status 15.
- `tests/test_imagewam_quant_linear.py` runs its FP8 tests on H100: the
  small case gives cosine 0.999242 for dynamic and static FP8 (the same
  value Thor measured), and every served ImageWAM shape (16 distinct
  `(M,N,K)`, merged single-stream `linear1`/`linear2`) gives cosine
  0.999291-0.999304, rel_l2 0.0373-0.0376 against `Fp16Linear` (random
  N(0,0.02) weight, N(0,1) input). The
  NN-vs-TN test skips here (NN unsupported) and runs on Thor.
- `sm110_check.sh`: builds; the Thor build exports the TN and NN symbols.
- End to end on H100 with the real checkpoint
  (`imagewam_e2e_official_compare.py`, `N_TASKS=3 FRAMES=0`): `fp8` gives
  `fr_vs_off` median 0.99845, MAE vs GT 0.20706 (official 0.20688).
  `fp8_static` runs too; its accuracy depends on the activation
  calibration (`opportunities.md` OPT-022).

Whether TN is as fast as NN on Thor is open; the Thor check is
`benchmarks/imagewam_fp8_layout_bench.py`.

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

# ISSUE-060

Status: resolved

Area: `ImageWAMTorchFrontendThor.set_prompt` prompt cache (`flash_rt/frontends/torch/imagewam_thor.py`)

## Observation

`set_prompt` returns early when `(prompt_text, context is not None)`
equals the cached key. Every call that passes a precomputed `context`
has the key `(None, True)`, so a second call with a different context
and mask is ignored: the previous task's context stays in the buffer
and `infer()` runs on it.

## Impact

Any caller that switches tasks through `set_prompt(context=...)` gets
actions conditioned on the wrong instruction, without an error.
`benchmarks/imagewam_e2e_official_compare.py`,
`benchmarks/imagewam_gate_fixture_generate.py` and
`tests/gate_imagewam_libero.py` work around it by setting
`frontend._current_prompt = None` before each new context. The
`prompt_text` path (live Qwen3) is not affected.

## Evidence

Source of the cache key: `cache_key = (prompt_text, context is not
None)`, compared before any copy. The workaround line is present in the
end-to-end script since commit `21a2080`.

## Hypotheses

The cache was written for the random-context and `prompt_text` paths,
where the key identifies the content. For a precomputed context the key
carries no content identity.

## Next Experiment

Skip the early return whenever `context` is given (a context copy is
cheap and the graph is captured only once), then remove the three
workarounds and confirm the gate's `fp16` result on the v1 fixture is
unchanged.

## Resolution

`set_prompt` returns early on its cache key only for the live-Qwen3 and
random paths; a precomputed `context` is applied on every call
(`flash_rt/frontends/torch/imagewam_thor.py`). The `_current_prompt`
reset is removed from `benchmarks/imagewam_e2e_official_compare.py`,
`benchmarks/imagewam_gate_fixture_generate.py` and
`tests/gate_imagewam_libero.py`.

Checks:

- `tests/test_imagewam_text_trim.py::test_new_context_of_the_same_length_is_applied`
  (both `text_trim` settings): a second context with the same mask
  replaces the context rows, changes the actions, and equals a fresh
  frontend given only that context, bit for bit.
- `tests/gate_imagewam_libero.py --precision fp16` on fixture v1 (H100),
  before (base code with the reset) and after (fix, no reset): the 40
  per-sample `vs_official`, `vs_fp16_reference` and MAE values are
  identical; `vs_official` median 0.998358, min 0.995532, mean
  `mae_vs_gt` 0.183642.

# ISSUE-061

Status: open (clock policy decided; baseline re-seed pending on Thor)

Area: regression-gate latency baseline for Thor `nvfp4` (`tests/fixtures/imagewam_gate/latency_baselines.json`)

## Observation

The seeded baseline, `infer()` P50 = 231.6 ms (`opportunities.md`
OPT-015), has no record of the Jetson power mode, the devfreq clock
state, or whether `use_fa4` was set when it was measured. The gate runs
the frontend with its defaults (`use_fa4=False`) and records the clock
state of every run (`flash_rt/hardware/jetson_clock_state.py`).

## Impact

If the baseline and a gate run differ in clock state or FA4, the
latency check compares different configurations. A run that differs by a few percent could pass or fail
for that reason alone. The margin is 5% (limit 243.18 ms).

## Evidence

OPT-014 and OPT-015 report P50 values without clock or FA4 details. The
40-call stability run before the `linear1` merge had a P50 of 243.5 ms
with a 243.2-247.3 ms range, against 236.9 ms in the precision table of
the same checklist.

## Hypotheses

Clock state and FA4 each move `infer()` P50 by a few percent on Thor.

## Next Experiment

On Thor, as the machine is (MAXN, DVFS-managed clocks), run the gate
for `nvfp4`, with and without `FLASHRT_THOR_FA4=1`. Re-seed the baseline
from that run and record its clock record and FA4 state in the
baseline's `source` field.

## Resolution

Clock policy decided: Thor is shared, and no benchmark or gate changes
its power mode or clocks (no `sudo`, `nvpmodel -m` or `jetson_clocks`).
Runs happen at the existing MAXN mode with DVFS-managed clocks. The
latency check records the clock state and never refuses dynamic clocks;
the baseline comes from a run in the same state. Re-seeding the baseline
from a Thor run is still pending.

# ISSUE-020

Status: resolved by the opt-in `text_trim=True`; the default stays untrimmed until Thor confirms it (ISSUE-080)

Area: text-token key padding in every attention call
(`flash_rt/hardware/thor/attn_backend.py` `ImageWAMAttnBackend`, both
sites; `flash_rt/models/imagewam/pipeline_thor.py`)

## Observation

The served pipeline attends to every one of the `x0` context rows. The
Qwen3 context is tokenized to a fixed 512 tokens with
`padding="max_length"` (`flash_rt/models/imagewam/text_encoder.py`).
LIBERO prompts have 16-31 valid tokens, so about 490 of the 512 text
rows are padding.

Official `_build_mot_attention_mask_flux2` in
`imagewam/models/backbones/imagewam.py` excludes the padded text keys
for every query row:

```python
mask[:, :, t0:r0] &= text_valid[:, None, :]
```

`infer_action_flux2` passes `text_attention_mask` to both of its calls,
the backbone prefill (`action_len=0`) and the action site
(`action_len>0`). `target_len=0` removes only the region mask.
FlashRT's `set_prompt()` reads `context_mask` only to place the
proprio row. At both attention sites FlashRT attends to the padded
keys and official does not.

## Impact

This one difference accounts for almost all of FlashRT's deviation
from official, and on some frames it is larger than official's own
seed-to-seed spread.

Measurement setup:

- 60 LIBERO frames: `libero_spatial`, `libero_goal` and `libero_10`,
  10 tasks x frames {0, 60} each.
- FlashRT fp16 against official bf16, seed 0, the same N(0,1) initial
  noise on both sides, H100.
- The official variants patch `_build_mot_attention_mask_flux2` to drop
  `text_attention_mask` at both calls, at the prefill call only, or at
  the action call only.

Action cosine:

| comparison | median | min | mean |
|---|---:|---:|---:|
| FlashRT vs official (masked, as shipped) | 0.99809 | 0.92997 | 0.99543 |
| FlashRT vs official with the text mask removed at both calls | 0.999979 | 0.99758 | 0.99990 |
| official masked vs official unmasked | 0.99812 | 0.92807 | 0.99557 |
| official, mask dropped at the prefill call only, vs masked | 0.99918 | 0.93622 | 0.99631 |
| official, mask dropped at the action call only, vs masked | 0.99825 | 0.98264 | 0.99794 |
| official seed 0 vs seed 1 (masked) | 0.99656 | 0.77926 | 0.98782 |

Per suite:

| suite | valid text tokens | FlashRT vs official median / min | FlashRT vs unmasked official median / min | official masked vs unmasked median / min |
|---|---|---|---|---|
| libero_spatial | 26-31 | 0.99840 / 0.99567 | 0.99998 / 0.99994 | 0.99839 / 0.99585 |
| libero_goal | 16-21 | 0.99681 / 0.92997 | 0.99998 / 0.99971 | 0.99690 / 0.92807 |
| libero_10 | 20-31 | 0.99860 / 0.96642 | 0.99998 / 0.99758 | 0.99860 / 0.97533 |

Four of 60 frames fall below 0.99 against official:

| frame | valid tokens | FlashRT vs official | FlashRT vs unmasked official | official masked vs unmasked | official seed 0 vs 1 |
|---|---:|---:|---:|---:|---:|
| libero_goal ep 0, frame 0 | 19 | 0.92997 | 0.99996 | 0.92807 | 0.99756 |
| libero_10 ep 0, frame 60 | 24 | 0.96642 | 0.99758 | 0.97533 | 0.77926 |
| libero_goal ep 300, frame 0 | 19 | 0.98105 | 0.99998 | 0.98091 | 0.99554 |
| libero_goal ep 338, frame 0 | 21 | 0.98137 | 0.99997 | 0.98133 | 0.98854 |

On libero_goal ep 0 frame 0, official's own seed spread is 0.99756, but
the mask alone moves official to 0.928.

Mean MAE of the 64-step chunk against ground truth barely moves:

| path | mean MAE |
|---|---:|
| official masked | 0.15873 |
| official unmasked | 0.15855 |
| FlashRT | 0.15868 |

The correlation between the valid-token count and the masked-vs-unmasked
cosine is 0.258.

## Evidence

- The upstream mask builder and both of its call sites, as quoted above.
- `ImageWAMAttnBackend.run()` has no key-mask input. With
  `use_real_mot_mask=True` both sites call unmasked
  `attention_qkv_fp16_perhead`.
- The measurements above. Removing the mask from official moves
  FlashRT's agreement from median 0.99809 / min 0.92997 to median
  0.999979 / min 0.99758.
- With proprio packing, padded keys form one contiguous row range,
  `[valid_count + 1, x0)`, between the proprio row and the image rows.

## Hypotheses

Under the official mask, padded text tokens are inert: no query reads
them. Their own queries still run, but nothing reads their outputs,
because the text rows' outputs do not feed the action path except
through K/V, and padded K/V are masked. If so, dropping the padded rows
from the sequence is exactly equivalent to the official mask, with no
mask kernel needed.

Supporting measurement, fp16, H100, 4 libero_goal frames including
ep 0 frame 0:

- FlashRT built with `x0 = n_valid + 1` (a0 and total adjusted:
  `a0 = x0 + 392`, `total = a0 + 64`).
- `set_prompt(context=context[:n_valid], context_mask=mask[:n_valid])`.
- Cosine against official masked: 0.999976-0.999989, including the
  frame where the full-length build gives 0.930.
- The shorter sequence also makes `infer()` faster.

## Next Experiment

Serve with the padded tokens dropped: size `x0` from the prompt's
valid-token count (`n_valid + 1` with proprio), so `a0` and `total`
follow, and capture per prompt length. The served frontend fixes `x0`
at construction today, so this needs either per-prompt construction or
buffers sized for the maximum length with a capture per length. Then
run `benchmarks/imagewam_e2e_official_compare.py` on
`libero_spatial`, `libero_goal` and `libero_10`, and compare
`fr_vs_off` against the table above; the expected median is
0.99998-level. Then re-measure `infer()` on Thor.

## Resolution

`ImageWAMTorchFrontendThor(text_trim=True)` (opportunities.md OPT-030)
builds the sequence from the valid tokens and the proprio row only
(`x0 = n_valid + 1`, `a0 = x0 + 392`, `total = a0 + 64`), with one CUDA
graph per distinct length over the max-size buffers. This is the
official masked math with no mask kernel.

H100, fp16 FlashRT vs official bf16, the same 60 frames
(`benchmarks/imagewam_e2e_official_compare.py`, `N_TASKS=10
FRAMES=0,60 SEEDS=0,1`, `TEXT_TRIM=0` / `1`), `fr_vs_off`:

| suite | valid tokens | untrimmed median / min | trimmed median / min |
|---|---|---:|---:|
| libero_spatial | 26-31 | 0.99840 / 0.99566 | 0.99998 / 0.99993 |
| libero_goal | 16-21 | 0.99680 / 0.92997 | 0.99998 / 0.99971 |
| libero_10 | 20-31 | 0.99860 / 0.96654 | 0.99998 / 0.99654 |
| all 60 frames | | 0.99809 / 0.92997 | 0.99998 / 0.99654 |

- Frames below 0.99: 4 untrimmed, 0 trimmed. libero_goal ep 0 frame 0
  goes from 0.92997 to 0.99998.
- The trimmed minimum is libero_10 ep 0 frame 60: 0.99654 at seed 0 and
  0.99986 at seed 1. Official's own seed 0 vs seed 1 cosine on that
  frame is 0.77926.
- Mean `mae_fr_vs_gt` moves onto official's own: libero_spatial
  0.18359 -> 0.18555 (official 0.18538), libero_goal 0.16201 -> 0.15958
  (0.15941), libero_10 0.13044 -> 0.13144 (0.13139).

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
- FA4 stays opt-in (`FLASHRT_THOR_FA4=1`, `use_fa4=True`,
  `use_fa4_mot=True`).

## Evidence

The H100 runs of the checklist scripts print `SKIP` for every NVFP4,
FP8 CUTLASS, and FA4 row. `tests/test_imagewam_fa4_backbone.py` skips
all three real-FA4 tests.

## Hypotheses

## Next Experiment

Run the Thor checklists in opportunities.md OPT-018 ("Thor check") and
OPT-019 ("Thor check").

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
path used one scale per half. The weight side is the same: both classes
quantize the whole merged `(K, N)` weight with one per-tensor FP8 scale,
where the split path had one scale for `attn_out_proj` and one for
`mlp_down`.

## Impact

Measured negligible at the GEMM level (Evidence). None of these
precisions is the shipped default (`nvfp4`, whose per-16-element block
scales make the merged and split operands identical). The end-to-end
merged vs split comparison for FP8 has not run: FP8 GEMMs fail on H100
until the ISSUE-001 TN-layout fix lands (calibration stream); after
that it can run on H100 as well as Thor.

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
(`benchmarks/imagewam_fusion_ab.py`), on Thor, or on H100 once
ISSUE-001's TN fix lands: merged vs split action cosine. Close this issue
if it matches `fp16`'s merged vs split (cos >= 0.9999); otherwise default
FP8 to the split path (`merge_linear2=False` for those precisions).

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
sizes. `pil_bilinear` is closer to the training transform than area
(token cosine 0.9963 vs 0.9926 at 512, 0.9957 vs 0.9793 at 256), but
neither reproduces training exactly: training resizes in float32 without
uint8 rounding. The official eval chain is as far from training as
`pil_bilinear` is.

## Next Experiment

Owner decision: make `vae_resize="pil_bilinear"` the served default
(same kernel cost; resize bit-exact to the official eval, token cosine
0.9999 to the official chain including its normalization). A training-
transform mode (float32 bilinear antialias) would be closer to training
still, but no reference eval uses it. A closed-loop LIBERO success-rate
A/B of `area` vs `pil_bilinear` at the eval's 256x256 rendering on Thor
would settle whether the difference matters beyond open loop.

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

# ISSUE-071

Status: resolved

Area: `ImageWAMTorchFrontendThor._graph` replay, final backbone residual
(`backbone_hidden` after the last single-stream block)

## Observation

On H100 (fp16, small random dims), a scratch parity script that ran the
Python pipeline eagerly (`pipeline_thor._single_stream_layer`,
`_double_stream_layer`, `imagewam_prefill`, `imagewam_denoise_loop`) and
the native pipeline eagerly, then replayed the Python graph once from a
restored input state, saw that replay's final `backbone_hidden` differ
from every other run by `max_abs` 0.0049 and 0.0035 in 2 of 7 script
runs. In the same replays the K/V caches, `Q_O` and `action_latent` were
bit-identical, and every later replay matched the eager result exactly.

## Impact

None on actions: after the last backbone layer, the residual is not read
by the denoise loop (it reads only the K/V caches). Bit-exact parity
checks that included `backbone_hidden` after a Python-graph replay failed
intermittently, so the native-pipeline test had dropped that buffer from
its native-graph vs Python-graph comparison.

## Evidence

- Targeted reproductions did not trigger it: 500 consecutive replays
  (0 mismatches); first replay after eager Python denoise, eager native
  denoise, or both (30 trials each, 0 mismatches).
- The difference is confined to the last block's gated residual update,
  i.e. to its `attn_out_proj` / `mlp_down` GEMMs, its `_add_inplace`, or
  its in-graph gate tensor (`_fuse_mod_group` output in the graph's
  private memory pool).

## Hypotheses

1. Memory in the graph's private pool that backs the last block's
   in-graph gate tensor was modified between capture and that replay.
2. A timing-dependent kernel choice or workspace effect in one of the
   last block's GEMMs under a co-tenant load.

## Next Experiment

Replay with the per-layer gate tensors hoisted out of the graph (the
native pipeline's precomputed modulation) and count mismatches over many
first replays after mixed eager work; if they disappear, hypothesis 1
holds.

## Resolution

Resolved 2026-09-18: a race in the test harness, not in the Python graph.

- Root cause: the snapshot that followed the Python-graph replay queued
  its `clone()` copies on the torch stream and returned without waiting
  for them. The next call, `NativeRuntime::capture()`, ran its eager
  warm-up on the native stream, which is created non-blocking and so is
  not ordered after the torch stream. The warm-up's first GEMM
  (`txt_in`, rows `[0, x0)` of `backbone_hidden`) could overwrite those
  rows before the copy read them. With small random weights the final
  residual is within a few bf16 ulps of the `txt_in` output, which is why
  the differences were small and confined to the text rows.
- Reproduction: with `backbone_hidden` back in
  `test_native_graph`'s comparison, the test failed in 15 of 24 runs on
  H100 (GPU shared with another process), with differences only in rows
  `[0, x0)`. With a second Python replay added, only the snapshot taken
  directly before `capture()` differed. A replica with no native work
  right after the snapshot never failed.
- Fix: `frt_imagewam_native_run` and `frt_imagewam_native_capture` now
  wait for all prior device work (`cudaDeviceSynchronize`) before they
  write the frontend's buffers, and the test's snapshot waits for its
  copies. Either change alone removed the failure: 0 of 16 runs failed
  (8 processes, both layer structures). With both
  changes, the native runtime and pipeline tests, including the
  restored comparison, passed in 10 of 10 runs.
- Consistent with independent evidence that the Python graph is
  deterministic: over 100 fresh-process runs of first versus later
  replays and eager runs, across scenarios and the base commit, showed 0
  mismatches, with one digest per structure across processes.

# ISSUE-080

Status: open (owner decision after Thor: served default of `text_trim`)

Area: `ImageWAMTorchFrontendThor(text_trim=...)`
(`flash_rt/frontends/torch/imagewam_thor.py`, opportunities.md OPT-030)

## Observation

`text_trim=True` removes FlashRT's largest deviation from official
(ISSUE-020) and shortens every backbone GEMM and attention call, but
it is opt-in. On H100 it is verified at `fp16`, `fp8` and `fp8_static`
(with a calibration file recorded trimmed). `nvfp4`, `e0m3_hadamard`,
`fp8_static_cutlass` and FA4 run on Thor only.

At seed 1 the trimmed fp16 libero_goal minimum vs official is
ep 114 frame 60, 0.99487-0.99510 over four H100 runs, where official's
own seed 0 vs seed 1 cosine is 0.93157 (seed 0 on the same frame:
0.99966-0.99971).

## Impact

The served default (`nvfp4`, untrimmed) keeps attending to about 490
padded text keys: action cosine vs official down to 0.930 at fp16 on
libero_goal, and about 30% more replay time than trimmed at fp16 on
H100.

Costs that come with the flag:

- The first `set_prompt` of each new text length captures a graph:
  0.6-2.0 s at fp16 on H100 (fp16 GEMM autotune of the new shapes is
  0.3-0.6 s of it; other precisions skip that part). Thor time is
  unmeasured. A cached length switches in about 10 ms.
- Each cached length holds one CUDA graph: 10-16 MiB on H100 (NVML,
  per process). The cache has no bound; LIBERO has 15 distinct lengths
  over its four suites, and the buffers allow up to 512.
- The regression gate's fixture v1 stores an untrimmed fp16 reference.
  Trimmed fp16 on that fixture (H100, 40 runs) measures vs official
  median 0.99998 / min 0.99992, but vs the stored fp16 reference median
  0.99837 / min 0.99579, below the fp16 bounds 0.999 / 0.995; mean MAE
  ratio to the reference 1.0115 (bound 1.02). Serving `text_trim=True`
  by default needs the fixture's fp16 reference regenerated with
  trimming (`benchmarks/imagewam_gate_fixture_generate.py`), a new
  fixture version.

## Evidence

opportunities.md OPT-030 (H100 numerics, speed, capture cost and
memory).

## Hypotheses

On Thor, trimming improves `nvfp4` agreement with official on every
suite by about as much as on fp16, and lowers `infer()` P50 because
the backbone (about 46% of `infer()`) runs about 420 instead of 905
rows.

## Next Experiment

Conditions for making `text_trim=True` the served default, all of them:

1. Thor `nvfp4` end-to-end compare with `TEXT_TRIM=0` and `1` on
   libero_spatial, libero_goal and libero_10, and the FA4 checks (plan.md
   Thor check, steps 2-4): trimmed agreement with official at or above
   untrimmed on every suite, `infer()` P50 lower.
2. The multi-length safety check on Thor at the served precision, FA4
   off and on (Thor check step 6,
   `tests/test_imagewam_text_trim_graph_safety.py`): every length
   bit-identical to a fresh single-length frontend, no weight-op tensor
   reallocated, no write into poisoned free memory.
3. A failed capture leaves no graph active (done: `set_prompt` raises,
   `infer()` refuses, cached lengths keep working).
4. Gate fixture v2 with a trimmed fp16 reference
   (`benchmarks/imagewam_gate_fixture_generate.py`): fixture v1's fp16
   reference is untrimmed and trimmed fp16 falls below its fp16 bounds
   (0.99837 / 0.99579 vs 0.999 / 0.995).
5. Every consumer of the frontend's graph and buffers uses the active
   per-length dims (opportunities.md OPT-030, "Constraints on consumers
   of a trimmed frontend"): the runtime surface and export with
   `text_trim` and the active `x0` in the setup identity and the graph
   re-adopted after the prompt verb, one native graph per length, and
   calibration files recorded trimmed (the file identity enforces the
   last). Until then `runtime_surface()`, `pipeline_resources()` and
   `export_model_runtime()` refuse a trimmed frontend.
6. Known prompt lengths captured at startup (`precapture_text_lengths`)
   and a bounded per-length cache, so no capture happens while serving
   and memory stays bounded.

Then the owner decides the default.

## Resolution

# ISSUE-081

Status: open

Area: `ImageWAMTorchFrontendThor.runtime_surface().view_shape`
(`flash_rt/frontends/torch/imagewam_thor.py`), consumed by
`flash_rt/models/imagewam/runtime_export.py` (`decode_image_views`,
the declared frame shape and the exported verb list)

## Observation

`runtime_surface()` reports

```python
view_shape=((2, 224, 224) if self._vae_stage is None else
            (self._vae_stage.spec.num_views, self._vae_stage.spec.in_h, self._vae_stage.spec.in_w))
```

so a frontend built with the VAE outside the CUDA graph (`vae_encoder="torch"`
or `"native"` with `vae_graph_input=None`, the `default` profile) declares
`(2, 224, 224)` whatever workload it was resolved for. `vae_graph_input`
carries the workload's own `(num_views, image_h, image_w)` when the VAE is in
the graph, so the hard-coded value is reached only on the outside-the-graph
path.

## Impact

`view_shape` is not descriptive: `decode_image_views` unpacks the ABI's image
payload with it, and `_input_shapes`/the exported verb list declare
`(*view_shape, 3)` uint8 frames. A deployment whose workload differs from
LIBERO's (a different camera count, or a per-view size other than 224x224)
therefore exports an ABI that decodes the wrong number of bytes per frame and
declares the wrong input shape, while `infer()` itself would take the views it
is handed. This is the deployment described by `THOR_CHECKLIST.md` section D,
whose workload fields are still to be entered.

## Evidence

- `flash_rt/frontends/torch/imagewam_thor.py`, `runtime_surface()`.
- `flash_rt/models/imagewam/runtime_export.py:100` (`decode_image_views(payload,
  view_shape)`), `:164` (`frames = decode_image_views(payload,
  self._surface.view_shape)`), `:196` (`frame_shape = (*surface.view_shape, 3)`),
  `:235` (`view_names(surface.view_shape[0])`).
- `ImageWAMWorkload.vae_graph_input()` returns the workload's
  `(num_views, image_h, image_w)` and is the only other source of the same
  geometry; `resolve_config` sets it only when the profile/override puts the
  VAE in the graph.

## Hypotheses

The constant predates the workload object (it was the LIBERO shape of the
frontend's own `set_prompt`/staging path) and was never revisited when
`view_shape` became the ABI's frame geometry.

## Next Experiment

Carry the geometry from the workload: a frontend built through
`from_config`/`load_imagewam` reports `workload.vae_graph_input()` as
`view_shape` on both paths (equal to the VAE stage's spec when the stage
exists), and a frontend built by the constructor with hand-passed dims keeps
today's value. Then check, on a target workload (section D), that the exported
runtime's `image_views` verb list and its declared frame shape follow the
workload, and that a LIBERO export is unchanged.

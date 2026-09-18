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

# ISSUE-060

Status: open

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

# ISSUE-061

Status: open

Area: regression-gate latency baseline for Thor `nvfp4` (`tests/fixtures/imagewam_gate/latency_baselines.json`)

## Observation

The seeded baseline, `infer()` P50 = 231.6 ms (`opportunities.md`
OPT-015), has no record of the Jetson power mode, the devfreq clock
state, or whether `use_fa4` was set when it was measured. The gate runs
the frontend with its defaults (`use_fa4=False`) and records the clock
state of every run (`flash_rt/hardware/jetson_clock_state.py`).

## Impact

If the baseline was measured with locked clocks or FA4 and a gate run
is not (or the reverse), the latency check compares different
configurations. A run that differs by a few percent could pass or fail
for that reason alone. The margin is 5% (limit 243.18 ms).

## Evidence

OPT-014 and OPT-015 report P50 values without clock or FA4 details. The
40-call stability run before the `linear1` merge had a P50 of 243.5 ms
with a 243.2-247.3 ms range, against 236.9 ms in the precision table of
the same checklist.

## Hypotheses

Clock state and FA4 each move `infer()` P50 by a few percent on Thor.

## Next Experiment

On Thor, run the gate for `nvfp4` twice: once as the machine is, and
once after `sudo nvpmodel -m 0 && sudo jetson_clocks`. Re-seed the
baseline from the locked-clock run and record its clock state in the
baseline's `source` field.

Owner decision: the latency check records the clock state in every
result but does not refuse unlocked clocks, unlike Pi0.5's
`machine_state()`, which raises. Whether an unlocked Thor run should
turn the latency verdict into `blocked` is open.

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

Status: open

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
checks that include `backbone_hidden` after a Python-graph replay can
fail intermittently; the native-pipeline test compares the native graph
with the eager Python run for that buffer.

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

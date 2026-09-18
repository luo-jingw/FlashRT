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
  value Thor measured), and every real ImageWAM shape (15 distinct
  `(M,N,K)`) gives cosine 0.999295-0.999306, rel_l2 0.0373-0.0376
  against `Fp16Linear` (random N(0,0.02) weight, N(0,1) input). The
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

# Pi0.5 and ImageWAM: Cross-Hardware Benchmark Status

## Inference test setup

Frozen target config (`CROSS_HW_BENCHMARK_PROTOCOL.md`): 3 cameras,
224x224, 10 ODE/denoise steps, action_dim=32, action_horizon=30, random
weights (no real checkpoint). Result schema:
`docs/benchmark_result_schema.json`.

Weights: `scripts/gen_synthetic_pi05_checkpoint.py` generates a
random-weight, shape-correct Pi0.5 checkpoint directory
(`docs/pi05_synthetic_checkpoint.md`). ImageWAM uses
`ImageWAMTorchFrontendThor(checkpoint_dir=None, dims_override={...})`,
which has always supported random weights. Both models' official
PyTorch reference implementations were pointed at the same
random-initialized weights (Pi0.5: `PI0Pytorch` loaded with
`strict=False`; ImageWAM: `action_dit_config.action_dim` set to 32 with
ActionDiT left randomly initialized, FLUX.2/AE/Qwen3 still loaded as
real weights since those are required to construct the official classes
at all, not part of ImageWAM's own policy weights). Random weights mean
there is no official reference to compare cosine against for any row
below (`correctness` is null throughout) — this campaign measures speed
only.

Known caveats, apply to every table below unless a row says otherwise:

- **OPT-035**: `Pi05TorchFrontendThor` has no `chunk_size`/`num_steps`
  constructor override (hardcoded to 10/10). Every Thor row for Pi0.5 is
  therefore measured at `action_horizon=10`, not the frozen target's 30
  — `Pi05TorchFrontendRtx` (RTX 5090 / Orin) does have both as real
  kwargs and hits the target shape exactly.
- The synthetic checkpoint generator omits two tensors OpenPI's own
  strict loader expects (`gemma_expert.lm_head.weight`,
  `paligemma.model.language_model.norm.weight`) — the official reference
  needs `strict=False` (missing keys left at their module's own random
  init) to load it at all. This does not affect FlashRT, which does not
  read `config.json` or use `strict` loading.
- OpenPI's own reference measured consistently slower on synthetic
  weights than on a real checkpoint at the same shape (roughly 13-25 ms
  slower) — not yet root-caused; suspected but unconfirmed cause is the
  two missing tensors keeping their module's own init dtype rather than
  the checkpoint's.
- ImageWAM's SM120 NVFP4 kernel requires K>=64; action_dim=32 does not
  meet that for the GEMM it applies to, so ImageWAM's NVFP4 row at the
  full target shape falls back to the real LIBERO action_dim (7) for
  that one precision only — see each table's own footnote.
- ImageWAM has a `text_trim` service default (trims text attention to
  the real valid token count rather than the full padded 512): a row
  measured with it off computes strictly more work than the served
  default and is not the number to quote as ImageWAM's real fastest
  config at this shape.

## RTX 5090

Commit `f7e4624`, CUDA event, warmup 20 / iters 100, random weights.

### Pi0.5 (3-view, action_horizon=30 — full target shape)

| Implementation | Precision | P50 (P10-P90) ms | matches_target |
|---|---|---:|:---:|
| Official (bf16 eager) | bf16 | 111.77 (111.56-112.08) | true |
| FlashRT | fp16 | 29.26 (29.23-32.32) | true |
| FlashRT | fp8 | 17.21 (17.19-17.25) | true |
| FlashRT | nvfp4 | — | blocked: SM120 NVFP4 not wired for Pi0.5 (deferred) |

### ImageWAM (3-view, action_horizon=30 — full target shape, `text_trim` on, 24 valid tokens)

| Implementation | Precision | P50 (P10-P90) ms | matches_target |
|---|---|---:|:---:|
| Official (bf16 eager, always computes the full 512 padded tokens) | bf16 | 489.5 | true |
| FlashRT | fp16 | 178.8 | true |
| FlashRT | fp16_cutlass | 183.5 | true |
| FlashRT | fp8 | 157.1 | true |
| FlashRT | fp8_static | 150.5 | true |
| FlashRT | fp8_static_cutlass | 126.6 | true |
| FlashRT | **nvfp4** | **116.9** | true |
| FlashRT | bf16 | — | blocked: FlashRT has no bf16 tier for ImageWAM |

Official's 489.5 ms computes strictly more text-attention work than
FlashRT's 116.9 ms (full 512 tokens vs `text_trim`'s 24 valid) — the two
numbers are not an equal-compute comparison. An equal-compute FlashRT
number (text_trim off, matching official's full-512 computation) was
also measured: nvfp4 154.2 ms.

ImageWAM's NVFP4 at the real LIBERO action_dim (7, not 32 — the SM120
kernel's K>=64 requirement forces this fallback) measured separately:
41.62 ms, `matches_target: false` (shape 3/224/10/**7**/30).

An earlier pass at this same target shape (superseded, do not use)
measured ImageWAM FlashRT at fp16 58.62 / fp8 37.96 ms and official at
128.65 ms — these predate the `text_trim`/equal-compute distinction
above and are not comparable to the corrected numbers in this table.

## Thor

Random weights, CUDA event, warmup 20 / iters 100.

### Pi0.5 (3-view, action_horizon=10 — OPT-035, not the target's 30)

| Implementation | Precision | P50 ms |
|---|---|---:|
| Official (bf16 eager), horizon=10 | bf16 | 285.6 |
| Official (bf16 eager), horizon=30 (off-target reference only) | bf16 | 298.4 (first run had a heavy tail, P90 573.8 ms; re-run stable at this value) |
| FlashRT + FA4 | fp16 | 80.9 |
| FlashRT + FA4 | fp8 | 47.9 |
| FlashRT + FA4 | **nvfp4** | **31.9** |

Compare FlashRT against official's **horizon=10 row (285.6 ms)** —
FlashRT Thor cannot run horizon=30 (OPT-035). These synthetic-weight
numbers land within ~5% of this project's earlier real-checkpoint Thor
measurements (fp16 85.84 / fp8 49.77 / nvfp4 31.74 ms,
`docs/pi05_thor_decoder_fp4_e2e.md`), which is the expected outcome for
a pure speed test (weight values should not change GEMM/kernel timing).

### ImageWAM (2-view, action_horizon=64 — real LIBERO shape, pre-existing; not yet measured at the 3-view/30-horizon target)

Canonical table: `docs/imagewam_results.md`. See that file for the full
precision ladder (official 456.7 ms, fp16 226.7, fp8 116.6, fp4 108.0
ms) — unchanged by this campaign, since it predates the target-shape
work and has not yet been re-run at 3-view/action_dim=32/horizon=30.

## Orin

Jetson Orin NX 16GB (SM87), CUDA event (wraps the whole `infer()`:
image upload, CUDA graph replay, action download; wall-clock P50 is
within 0.1 ms of the CUDA-event number), warmup 20 / iters 100. MAXN,
GPU clock **not locked** (DVFS, up to 918 MHz) -- P10-P90 spread is
still only 2-3 ms, so the clock stayed effectively stable during each
run despite not being pinned. Output shape confirmed `(30, 32)`.

Unlike every other table in this document: **real weights**
(`stack-cube-eef-24k` checkpoint, real recorded 3-camera frames), only
`action_horizon` overridden to 30 -- not the random-weights synthetic
checkpoint. `matches_target` below means the shape matches (3-view,
224x224, 10-step, action_dim=32, horizon=30); it does not mean random
weights. No official row: this machine has no working OpenPI PyTorch
install (FlashRT-only was also what was asked for this pass).

### Pi0.5 (3-view, action_horizon=30 — full target shape, real weights)

| Precision | P50 (P10-P90) ms | matches_target | status |
|---|---:|:---:|---|
| bf16 | 538.62 (537.42-540.46) | true | ok |
| fp16 | — | — | blocked: this build's FA2 only compiled bf16 |
| fp8 | — | — | blocked: SM87 has no FP8 tensor cores |
| nvfp4 | — | — | blocked: SM87 has no FP4 tensor cores |
| int8 | 421.04 (419.89-422.28) | true | ok |
| int8_hadamard | 467.23 (466.16-468.58) | true | ok |

`int8`/`int8_hadamard` here are this run's own precision names; they
have not yet been reconciled against `docs/deployment_orin.md`'s
existing `cache_frames=1/2` INT8 convention (same underlying SM87 INT8
path, different naming so far -- worth checking they refer to the same
thing before quoting both documents side by side). Sanity check against
RTX 5090 (this doc's own table above, synthetic weights): 5090 fp16 29
ms / fp8 17 ms vs Orin bf16 539 ms is roughly an 18x gap, consistent
with the two GPUs' compute difference.

ImageWAM: not yet measured on Orin.

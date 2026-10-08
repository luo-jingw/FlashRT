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
| FlashRT | nvfp4 (encoder) | see below | true |

The nvfp4 row is `Pi05TorchFrontendRtx(use_fp4_encoder=True)`: the 18
Gemma encoder layers' qkv/o/gate_up/down run as SM120 NVFP4 W4A4 GEMMs;
vision, the vision projector and the decoder stay FP8. It was measured
later than the rows above, with the SM clock unlocked (2.40-2.87 GHz
observed), so it is timed interleaved with an fp8 frontend in one
process rather than against the 17.21 ms row:

| Measurement (same process, interleaved) | fp8 | nvfp4 encoder |
|---|---:|---:|
| Encoder stage, graph replay P50 (round robin, 200 samples) | 6.99 ms | 5.70 ms |
| `infer()` P50, run 1 | 17.54 ms | 16.42 ms |
| `infer()` P50, run 2 (`benchmark_results/pi05_flashrt_rtx5090_nvfp4.json`) | 19.03 ms | 18.21 ms |

Cosine of the 32-dim raw actions, nvfp4 encoder vs fp8, identical noise,
synthetic weights: 0.9995. The decoder stays FP8 because at its M=30
every SM120 NVFP4 tile is slower than FP8 (per-GEMM table in
`opportunities.md` OPT-036).

### ImageWAM (3-view, action_horizon=30 — full target shape)

| Implementation | Precision | P50 (P10-P90) ms | matches_target |
|---|---|---:|:---:|
| Official (bf16 eager) | bf16 | 128.65 (128.31-129.02) | true |
| FlashRT | fp16 | 58.62 (58.61-58.64) | true |
| FlashRT | fp8 | 37.96 (37.92-37.99) | true |
| FlashRT | nvfp4 | 41.62 (41.60-41.63) | false: SM120 NVFP4's K>=64 floor forces this row back to the real LIBERO action_dim (7), shape 3/224/10/**7**/30 |

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
Re-confirmed again on commit `fb9ed3c` (after the OPT-032 candidate 5
RMSNorm fix, which does not touch Pi0.5's own kernels): fp16 85.0 / fp8
49.2 / nvfp4 33.6 ms, synthetic weights -- 1.2-4 ms off both this row
and the real-checkpoint numbers above, treated as ordinary run-to-run
variation, not investigated further.

### ImageWAM (3-view, action_horizon=30 — full target shape, `text_trim` on, 24 valid tokens)

Re-measured on commit `fb9ed3c` with OPT-032 candidate 5
(`dims_override={"vec_rms_norm": True}`, not yet reachable through
`load_imagewam`'s named profiles) on top of every precision below the
official row, random weights:

| Implementation | Precision | P50 ms, before | P50 ms, `vec_rms_norm=True` | change | matches_target |
|---|---|---:|---:|---:|:---:|
| Official (bf16 eager, always computes the full 512 padded tokens) | bf16 | 489.5 | not re-run (no FlashRT code on this path) | — | true |
| FlashRT | fp16 | 178.8 | 161.7 | -17.1 | true |
| FlashRT | fp16_cutlass | 183.5 | 161.5 | -22.0 | true |
| FlashRT | fp8 | 157.1 | 138.1 | -19.0 | true |
| FlashRT | fp8_static | 150.5 | 130.3 | -20.2 | true |
| FlashRT | fp8_static_cutlass | 126.6 | 106.7 | -19.9 | true |
| FlashRT | **nvfp4** | 116.9 | **98.0** | -18.9 | true |
| FlashRT | bf16 | — | — | — | blocked: FlashRT has no bf16 tier for ImageWAM |

Every precision drops by roughly the same ~17-22 ms, consistent with
removing a fixed per-`infer()` RMSNorm cost that does not depend on the
GEMM precision (OPT-032 candidate 5's own Thor kernel-level breakdown:
23.008 -> 2.733 ms). `fp16`'s first run measured 189.3 ms (an outlier --
a same-commit re-check with the flag off and on separately gave 179.8
and 161.7 ms, confirming the flag itself speeds things up normally;
179.8 ms is closer to the `fp16_cutlass`/`fp8` family's own before-row
than 178.8 ms is, so the outlier is attributed to that one construction,
not to this table's "before" column). The old (pre-fix) numbers are
kept in `logs/0928_crosshw/side/` (`_trim24_novec` suffix) for
reference.

Official's 489.5 ms computes strictly more text-attention work than
FlashRT's 98.0 ms (full 512 tokens vs `text_trim`'s 24 valid) — the two
numbers are not an equal-compute comparison. An equal-compute FlashRT
number (`text_trim` off, matching official's full-512 computation,
pre-RMSNorm-fix) was also measured: nvfp4 154.2 ms.

Cross-check against this project's earlier real-checkpoint Thor number
at ImageWAM's native shape (2-view, horizon=64, `text_trim` on, FA4 both
sites, FLUX.2 VAE in-graph — `docs/imagewam_results.md`'s `0921d_final`,
108.0 ms): reproducing that exact native-shape config with random
weights and `vec_rms_norm=True` gives 87.3 ms, matching the real-checkpoint
`vec_rms_norm=True` confirmation (86.4 ms, OPT-032 candidate 5's own
Thor entry) within noise -- the random-weight methodology continues to
track the real-checkpoint one after this fix. Extending only the camera
count to 3 accounts for the remaining gap to this table's 98.0 ms.

### ImageWAM (2-view, action_horizon=64 — real LIBERO shape, unchanged by this campaign)

`libero` workload, 2 views, text 512 tokens (16-31 valid), horizon 64,
action_dim 7, proprio 8, shift 5.0, 10 denoise steps. Commit `824f058`,
2026-09-21, warmup 20 / iters 100, real trained checkpoint
(ImageWAM-FLUX.2-4B-LIBERO). This is ImageWAM's native LIBERO shape, not
the campaign's 3-view/action_dim=32/horizon=30 target above.

| Row | P50 ms | P10-P90 ms | vs official | cos vs official (median/min) | MAE vs GT |
|---|---:|---:|---:|---:|---:|
| Official (torch, bf16) | 456.7 | 456.1-457.6 | 1.00x | — | — |
| Ours fp16 | 226.7 | 226.3-227.7 | 2.01x | 0.99998 / 0.99994 | 0.1856 |
| Ours fp8 | 116.6 | 116.5-116.7 | 3.92x | 0.99994 / 0.99989 | 0.1859 |
| Ours fp4 | 108.0 | 108.0-108.2 | 4.23x | 0.99936 / 0.99885 | 0.1861 |
| Ours int8/int4 (GEMM-only) | — | — | — | — | — |

Official's own end-of-session repeat measured 377.0 ms (-17.4%) in the
same session, so its row is not clock-bracketed as tightly as the
`Ours` rows; ImageWAM official reference MAE is 0.1855. int8/int4 are
not measured: the built extension has no int8/int4 fp16-out kernel, and
it was decided not to add one (`opportunities.md`).

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

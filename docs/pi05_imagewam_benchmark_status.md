# Pi0.5 and ImageWAM: Thor and RTX 5090 Benchmark Status

Reading rule: a row is only comparable to other rows in the same table.
Never compare an absolute millisecond number across models (Pi0.5 vs
ImageWAM) — their inputs are not the same shape (see "Cross-model
comparability" below) — and never across a table's own clock protocol (wall
vs CUDA event) or camera-view count without checking the table's own
header first. Full derivations, kernel-level breakdowns, and open items
live in `docs/pi05_thor_decoder_fp4_e2e.md`, `docs/imagewam_results.md`,
`issues.md` (ISSUE-092), and `opportunities.md` (OPT-032, OPT-033).

## Cross-model comparability

Pi0.5 and ImageWAM are not the same workload even at matching camera
resolution (both use 224x224). The two dimensions that differ and change
how latency is spent:

| Dimension | Pi0.5 | ImageWAM (`libero` workload) |
|---|---|---|
| Cameras (aligned rows) | 2 (matches ImageWAM's own I/O) or 3 (Pi0.5's own native LIBERO input) | 2 |
| Action horizon | H=10 | H=64 |
| Text tokens | ~13 (fixed prompt) | 512 padded, 16-31 valid, trimmed + precomputed (negligible actual cost) |
| Model | Pi0.5 (PaliGemma-2B-class VLA) | FLUX.2 DiT + Qwen3-4B (world model) |

Only Pi0.5's own 2-view rows (below) are camera-count-aligned with
ImageWAM. Horizon (10 vs 64) and the underlying model are still different
— a "matched input" comparison controls for camera count and denoise-step
count only, not for what the two models are.

## Pi0.5 — Thor

### 3-view (Pi0.5's own native LIBERO input; not camera-aligned with ImageWAM)

`load_model()`, 2026-09-23 fill-in, warmup 20 / iters 100, same
`pi05_libero` weights and an 8-observation fixture, 10-step, 13 tokens.

| Row | P50 ms (P10-P90) | vs this session's fp8 | vs official |
|---|---:|---:|---:|
| Official (PyTorch eager bf16) | 267.97 (267.15-271.00) | 0.19x | 1.00x |
| FlashRT fp16 + FA4 | 85.84 (85.45-86.30) | 0.58x | 3.12x |
| FlashRT fp8 + FA4 | 49.77 (49.67-49.90) | 1.00x | 5.38x |
| FlashRT nvfp4 + FA4 | 31.74 (2026-08-05, not re-measured this session) | 1.54x vs the 2026-08-05 fp8 (49.02 ms) | — |

OpenPI's own JAX reference does not run on this Thor at all (`jax==0.5.3`
does not recognize compute capability 11.0); the runnable official baseline
is OpenPI's PyTorch eager reference, not the JAX original. The re-measured
fp8 (49.77 ms) matches the 2026-08-05 value (49.02 ms) within 0.8 ms.

### 2-view (camera-count-aligned with ImageWAM; earlier FlashRT round, 10-step)

| Row | P50 ms | vs fp16 |
|---|---:|---:|
| FlashRT fp8 | 38.70 | — |
| FlashRT nvfp4 + FA4 | 27.17 | — |

## Pi0.5 — RTX 5090

### 3-view (Pi0.5's own native LIBERO input; not camera-aligned with ImageWAM)

Commit `7a68a1c`, 10-step, warmup 10 / iters 100, wall clock.

| Row | P50 ms (P10-P90) | vs fp16 |
|---|---:|---:|
| Official OpenPI | — | no 3-view number: OpenPI's own LIBERO harness only feeds 2 cameras (zero-padding the right wrist), not comparable to FlashRT's 3-real-camera row |
| FlashRT fp16 | 28.76 (28.73-28.80) | 1.00x |
| FlashRT fp8 | 16.84 (16.82-17.08) | 1.71x |
| FlashRT fp4 | — | not wired, deferred |

fp16-to-fp8 ratio (1.71x) matches Thor's same-session ratio (85.84/49.77 =
1.73x) within noise — the same relative FP8 win holds on both
architectures despite the ~3x difference in absolute latency.

One open correctness item, independent of the speed numbers above: fp8 vs
fp16 cosine collapses to 0.42-0.61 on one of five tested seeds
(`encoder_ffn_down_w_14/15/16`, an outlier-activation-channel effect —
ISSUE-092, open).

### 2-view (camera-count-aligned with ImageWAM), wall clock, warmup 10 / iters 100

| Row | P50 ms (P10-P90) | vs official |
|---|---:|---:|
| Official OpenPI | 43.88 (43.83-44.03) | 1.00x |
| FlashRT fp16 | 23.28 (23.26-23.32) | 1.88x |
| FlashRT fp8 | 14.85 (14.82-14.90) | 2.96x |
| FlashRT fp4 | — | not wired, deferred |

This 2-view FlashRT row predates `7a68a1c` — it has not been cleanly
rebuilt and re-measured on the same commit as the 3-view row and
ImageWAM's own `7a68a1c` numbers above. Official OpenPI's 43.88 ms is on
`7a68a1c`-era clocks. Treat the FlashRT fp16/fp8 numbers in this row as
indicative, not commit-pinned; re-run on `7a68a1c` before quoting them
alongside ImageWAM's RTX 5090 table in the same document.

## ImageWAM — Thor

`libero` workload, 2 views, H=64, 10 denoise steps. Canonical per-precision
table (random weights, no VAE/proprio): `docs/imagewam_results.md`
(machine-generated, commit `824f058`, 2026-09-21):

| Row | P50 ms (P10-P90) | vs official | cos vs official (median/min) |
|---|---:|---:|---:|
| Official (torch, bf16) | 456.7 (456.1-457.6) | 1.00x | — |
| Ours fp16 | 226.7 (226.3-227.7) | 2.01x | 0.99998 / 0.99994 |
| Ours fp8 | 116.6 (116.5-116.7) | 3.92x | 0.99994 / 0.99989 |
| Ours fp4 | 108.0 (108.0-108.2) | 4.23x | 0.99936 / 0.99885 |
| Ours int8/int4 (GEMM-only) | — | — | not measured: the built extension has no int8/int4 fp16-out kernel; decided not to add one |

Three other configurations exist and must not be merged into the table
above — same model, different serving configuration:

| Configuration | fp16 | fp8 | nvfp4 | Note |
|---|---:|---:|---:|---|
| Gate baseline (`use_fa4=false`, no `text_trim`/VAE-in-graph) | 275.2 | 228.0 | 202.3 | `SUMMARY.txt`'s 268/223/203 is this same configuration |
| Service `default` (trim + FA4 both sites + native VAE in graph), real checkpoint, 24 valid tokens | — | — | 108.03 (`0921d_final`) / 106.3-106.4 (`0922i` regression, per-suite) | the number to quote as "ImageWAM's fastest served Thor config" |
| `3x256x256` target workload | — | — | 121.92 (`0921`, `profile=default`) | different input shape than the `libero` workload above; do not merge into either table |

## ImageWAM — RTX 5090

Commit `7a68a1c`, dual 224x224, 10-step, CUDA event, warmup 5 / iters 20.

| Row | P50 ms (P10-P90) | vs official | cos vs official (median/min) |
|---|---:|---:|---:|
| official (bf16) | 119.69 (119.20-120.03) | 1.00x | — |
| Ours fp16 | 59.65 (59.64-59.66) | 2.01x | 0.97923 / 0.97866 |
| Ours fp8 | 42.37 (42.36-42.41) | 2.82x | 0.97911 / 0.97852 |
| Ours nvfp4 | 46.47 (46.45-46.48) | 2.58x | 0.97859 / 0.97794 |

fp8 is the recommended default on this hardware. NVFP4 is bit-correct but
4.10 ms slower than fp8 (root-caused to the SM120 CUTLASS kernel itself at
ImageWAM's GEMM shapes, not to cast/quantize overhead or variant selection
— OPT-033, not started). GEMM-variant sweep at the same commit (not the
main table): `pingpong` (default) 46.33 ms e2e / 15.54 ms GEMM-only,
`plain` 46.10 / 15.29, `widen` 54.67 / 23.87.

## Matched-input cross-model view (2 cameras, 10 denoise steps, best per-hardware config)

Controls for camera count and step count only — see "Cross-model
comparability" above for what this does *not* control for (action
horizon, model identity).

| | Cameras | Horizon | Best measured P50 |
|---|---|---:|---:|
| Pi0.5, FlashRT nvfp4+FA4 (Thor) | 2x224 | 10 | 27.17 ms |
| Pi0.5, FlashRT nvfp4+FA4 (RTX 5090) | — | — | not measured at nvfp4 (fp4 not wired on RTX 5090) |
| ImageWAM, FlashRT nvfp4 service default (Thor) | 2x224 | 64 | 108.0 ms |
| ImageWAM, FlashRT fp8 (RTX 5090) | 2x224 | 64 | 42.37 ms |

## Orin

No aligned data for either model:

- **Pi0.5**: real Orin data exists (`docs/deployment_orin.md`, INT8 W8A8,
  added 2026-05-19) but on a different checkpoint (`pi05_droid`, not
  `pi05_libero`), different camera count (2-view, matches ImageWAM's count
  but not the 3-view Thor/RTX5090 rows above), and using a frame-caching
  technique (`cache_frames=2`) not present in any other row in this
  document. Not comparable to the Thor/RTX 5090 rows without a fresh
  `pi05_libero`, matched-clock re-run.
- **ImageWAM**: no real Orin deployment exists at all. `opportunities.md`
  OPT-007's Orin references are all analysis on Ada-class hardware about
  what Orin's own SM80-family INT4/INT8 CUTLASS path would likely do,
  never a real Orin measurement.

## Remaining gaps

| Table | Missing row | Status |
|---|---|---|
| Pi0.5 x RTX 5090 (3-view) | Official OpenPI | cannot be produced: OpenPI's own LIBERO harness only supports 2 cameras |
| Pi0.5 x RTX 5090 (2-view) | FlashRT fp16/fp8 on `7a68a1c` | needs a clean rebuild + re-measure; current numbers predate that commit |
| Pi0.5 x RTX 5090 | FlashRT fp4 | deferred, not wired |
| ImageWAM x Thor | int8/int4 GEMM-only | decided not to add the missing kernel |
| Pi0.5 x Orin | `pi05_libero`, 3-view or 2-view, matched clocks | not attempted; current Orin data uses a different checkpoint and camera count |
| ImageWAM x Orin | everything | no real deployment attempted |

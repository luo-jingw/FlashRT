# Pi0.5 and ImageWAM: Thor and RTX 5090 Benchmark Status

Four tables, one per (model, hardware) pair. Each row within a table comes
from the same commit and the same measurement session; do not compare
absolute milliseconds across tables. Full derivations, kernel-level
breakdowns, and open items live in `docs/pi05_thor_decoder_fp4_e2e.md`,
`docs/imagewam_results.md`, `issues.md` (ISSUE-092), and `opportunities.md`
(OPT-032, OPT-033).

## Pi0.5 — Thor

`load_model()`, 3-view, 2026-09-23 fill-in, warmup 20 / iters 100, same
`pi05_libero` weights and an 8-observation fixture.

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

## Pi0.5 — RTX 5090

Commit `7a68a1c`, 3-view / 10-step, warmup 10 / iters 100.

| Row | P50 ms (P10-P90) | vs fp16 |
|---|---:|---:|
| Official OpenPI | — | no 3-view number: OpenPI's own LIBERO harness only feeds 2 cameras (zero-padding the right wrist), not comparable to FlashRT's 3-real-camera row |
| FlashRT fp16 | 28.76 (28.73-28.80) | 1.00x |
| FlashRT fp8 | 16.84 (16.82-17.08) | 1.71x |
| FlashRT fp4 | — | not wired, deferred |

Official OpenPI at 2-view, same clocks, for scale only (not a same-harness
baseline for the row above): P50 43.88 ms (43.83-44.03).

fp16-to-fp8 ratio (1.71x) matches Thor's same-session ratio (85.84/49.77 =
1.73x) within noise — the same relative FP8 win holds on both
architectures despite the ~3x difference in absolute latency.

One open correctness item, independent of the speed numbers above: fp8 vs
fp16 cosine collapses to 0.42-0.61 on one of five tested seeds
(`encoder_ffn_down_w_14/15/16`, an outlier-activation-channel effect —
ISSUE-092, open).

## ImageWAM — Thor

`libero` workload, 2 views, 10 denoise steps, commit `824f058`, 2026-09-21.
Canonical table: `docs/imagewam_results.md` (machine-generated).

| Row | P50 ms (P10-P90) | vs official | cos vs official (median/min) |
|---|---:|---:|---:|
| Official (torch, bf16) | 456.7 (456.1-457.6) | 1.00x | — |
| Ours fp16 | 226.7 (226.3-227.7) | 2.01x | 0.99998 / 0.99994 |
| Ours fp8 | 116.6 (116.5-116.7) | 3.92x | 0.99994 / 0.99989 |
| Ours fp4 | 108.0 (108.0-108.2) | 4.23x | 0.99936 / 0.99885 |
| Ours int8/int4 (GEMM-only) | — | — | not measured: the built extension has no int8/int4 fp16-out kernel; decided not to add one |

## ImageWAM — RTX 5090

Commit `7a68a1c`, dual 224x224, 10-step, warmup 5 / iters 20.

| Row | P50 ms (P10-P90) | vs official | cos vs official (median/min) |
|---|---:|---:|---:|
| official (bf16) | 119.69 (119.20-120.03) | 1.00x | — |
| Ours fp16 | 59.65 (59.64-59.66) | 2.01x | 0.97923 / 0.97866 |
| Ours fp8 | 42.37 (42.36-42.41) | 2.82x | 0.97911 / 0.97852 |
| Ours nvfp4 | 46.47 (46.45-46.48) | 2.58x | 0.97859 / 0.97794 |

All four rows are complete; fp8 is the recommended default on this
hardware. NVFP4 is bit-correct but 4.10 ms slower than fp8 (root-caused to
the SM120 CUTLASS kernel itself at ImageWAM's GEMM shapes, not to
cast/quantize overhead or variant selection — OPT-033, not started).

## Remaining gaps

Everything not filled in above is a decision or a hardware limitation, not
an outstanding measurement:

| Table | Missing row | Status |
|---|---|---|
| Pi0.5 x RTX 5090 | Official OpenPI, 3-view | cannot be produced: OpenPI's own LIBERO harness only supports 2 cameras |
| Pi0.5 x RTX 5090 | FlashRT fp4 | deferred, not wired |
| ImageWAM x Thor | int8/int4 GEMM-only | decided not to add the missing kernel |

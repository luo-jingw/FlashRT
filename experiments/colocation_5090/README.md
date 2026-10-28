# Planner + executor co-location on one RTX 5090

Question: when a slow VLM planner shares one GPU with a real-time WAM executor, how much executor latency and how many
missed control deadlines does each sharing mode cost, and what limits the number of models / sessions on one device?

## Setup

| item | value |
|---|---|
| GPU | RTX 5090 32 GB (31.35 GiB usable), driver clocks unmanaged, no other GPU process |
| executor | ImageWAM (FLUX.2 klein 4B + ActionDiT), FlashRT fp8, LIBERO workload; every tick runs `stage_inputs` + CUDA-graph replay (VAE encode, prefill, 10-step denoise) |
| executor schedule | period 50 ms (20 Hz); latency = finish − release; an overrun delays the next release (`ticks` then drops below 1201 per 60 s) |
| planner | Qwen3-VL-2B-Instruct bf16, HF `generate`, one 448 px image, 32 new tokens per call |
| planner context | ~260 tokens (short) or ~16.3 k tokens (long, padded text) |
| planner rate | 0.5 / 1 / 2 Hz target, or continuous (back-to-back) |
| window | 60 s per cell, after warm-up |
| GPU telemetry | `nvidia-smi --loop-ms=100`: utilization.gpu, utilization.memory (DRAM), clocks, power |

Sharing modes:

| mode | mechanism |
|---|---|
| `serial` | one thread; a due planner call runs to completion before the next executor tick |
| `process` | planner in a second process; the driver time-slices the two contexts |
| `prio` | planner in a second thread of the executor's process; executor on a stream with priority −1 |
| `mpsP` | planner in a second process under an MPS daemon; planner `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` = P, executor = 100 − P (`mps100`: no SM limit on either) |

Files: `colocation_bench.py` (one cell), `planner_qwen3vl.py` (planner, R2 sweep), `run_r1.sh` + `run_r1b.sh` (R1 grid),
`summarize_r1.py` → `r1_table.md`, `session_switch.py` + `run_r3.sh` (R3), raw results in `r1/`, `r3/`, `r2_planner_cost.json`.

## R2: planner cost alone

| prompt tokens (1 image) | prefill ms | decode ms / token | call ms (32 tokens) | peak memory GB |
|---|---|---|---|---|
| 259 | 20.5 | 9.0 | 300 | 4.3 |
| 1 261 | 35.6 | 9.0 | 316 | 4.5 |
| 4 261 | 96.7 | 9.3 | 384 | 5.0 |
| 16 261 | 500 | 9.2 | 787 | 7.0 |
| 32 261 | 1 214 | 10.8 | 1 549 | 9.7 |

Decode at ~9 ms / token is CPU-launch-bound (eager HF loop); 2 or 4 images add < 10 % at the same text length.

## R1: executor under a co-located planner

Full 44-cell table (R1 + R4): `r1_table.md`. Condensed (executor latency in ms, miss = latency above the 50 ms period):

| planner context | mode | planner rate | exec p50 | exec p99 | miss % | planner calls/s achieved | planner call ms |
|---|---|---|---|---|---|---|---|
| — | isolated | — | 24.8 | 25.1 | 0 | — | — |
| short | serial | 0.5 / 2 Hz | 25.0 / 25.0 | 325 / 333 | 2.9 / 21.9 | 0.5 / 2.0 | 300 |
| short | serial | continuous | 319 | 335 | 100 | 3.1 | 295 |
| short | process | 1 Hz / continuous | 25.2 / 40.7 | 41.2 / 42.9 | 0 / 0 | 1.0 / 2.4 | 435 / 423 |
| short | prio | 1 Hz / continuous | 25.2 / 30.4 | 39.1 / 35.0 | 0 / 0 | 1.0 / 3.2 | 324 / 309 |
| short | mps10 | continuous | 26.6 | 26.9 | 0 | 2.25 | 446 |
| short | mps15 | continuous | 27.6 | 27.9 | 0 | 2.72 | 369 |
| short | mps25 | 1 Hz / continuous | 27.6 / 29.2 | 29.4 / 29.5 | 0 / 0 | 1.0 / 3.1 | 323 / 321 |
| short | mps50 | 1 Hz / continuous | 31.9 / 35.2 | 35.6 / 35.5 | 0 / 0 | 1.0 / 3.3 | 294 / 303 |
| short | mps100 | 1 Hz / continuous | 25.7 / 31.2 | 34.4 / 34.9 | 0 / 0 | 1.0 / 3.2 | 320 / 311 |
| long | serial | 0.5 / 1 Hz | 25.1 / 25.0 | 807 / 824 | 4.0 / 21.6 | 0.5 / 1.0 | 781 / 793 |
| long | serial | continuous | 809 | 820 | 100 | 1.25 | 784 |
| long | process | 0.5 Hz / continuous | 45.7 / 54.5 | 55.1 / 55.1 | 44.9 / 60.0 | 0.52 / 0.65 | 1 553 / 1 563 |
| long | prio | 0.5 Hz / continuous | 25.1 / 39.3 | 287 / 294 | 6.4 / 30.4 | 0.5 / 1.18 | 851 / 837 |
| long | mps10 | continuous | 26.3 | 27.2 | 0 | 0.22 | 4 850 |
| long | mps15 | continuous | 27.4 | 28.6 | 0 | 0.30 | 3 347 |
| long | mps25 | continuous | 29.0 | 31.5 | 0 | 0.50 | 2 065 |
| long | mps50 | continuous | 36.5 | 42.1 | 0 | 0.83 | 1 202 |
| long | mps100 | continuous | 39.8 | 304 | 30.4 | 1.18 | 850 |

Controls:

| cell | exec p50 / p99 ms |
|---|---|
| isolated, start of R1 / end of R1 | 24.8 / 25.1, 25.1 / 25.3 |
| executor alone under MPS with 90 / 85 / 75 / 50 % of SMs | 25.5 / 25.8, 26.2 / 26.4, 27.5 / 27.8, 31.8 / 32.1 |
| short prio continuous, text encoder resident / freed | 30.4 / 35.0, 30.0 / 31.4 |

Memory: the executor reserves 23.4 GB with its Qwen3-4B text encoder resident and 15.3 GB after freeing it. Executor
(text encoder resident) + long-context planner does not fit in 32 GB: the planner process fails with CUDA OOM at its
first 16 k prefill. All long-context cells and all MPS cells therefore run with the executor's text encoder freed after
`set_prompt` (`DROP_TEXT_ENCODER=1`); a session that changes its instruction would need it back.

## Observations

1. **Serial sharing fails as soon as a planner call is longer than the slack.** The executor cannot run during a call
   (300 ms short, 780 ms long), so the miss rate grows with planner rate (short: 2.9 % at 0.5 Hz → 21.9 % at 2 Hz).
2. **Time-slicing (`process`) bounds the executor's tail but stretches it.** Short context: p99 ≤ 43 ms, no misses, but p50
   rises 25 → 41 ms with a busy planner. Long context: the executor settles at ~54.5 ms, 45–60 % of ticks miss, and the
   planner call itself doubles (787 → 1 560 ms).
3. **Stream priority and MPS without an SM limit do not protect the executor from a long prefill.** With the long-context
   planner the executor's p99 is 287–304 ms in both `prio` (same process) and `mps100` (separate process), so the stall
   is GPU-side, not the Python GIL. It is not one long kernel (R4 trace below): the 16 k prefill is 2 434 kernels over
   543 ms, the longest 42.9 ms. The executor's tick is 5 240 kernels, and during the prefill it waits a little at
   almost every one of them (R4 overlap trace below): span 307 ms, of which its own kernels run 34.7 ms and 272 ms is
   spent between them, the longest single gap 4.75 ms. With the short planner both modes keep p99 ≤ 39 ms.
4. **An MPS SM limit gives the executor a hard bound, paid for in planner latency.** Long context, continuous planner:

   | planner SM limit | executor p99 ms | miss % | planner call ms |
   |---|---|---|---|
   | none (`mps100`) | 304 | 30.4 | 850 |
   | 50 % | 42.1 | 0 | 1 202 |
   | 25 % | 31.5 | 0 | 2 065 |
   | 15 % | 28.6 | 0 | 3 347 |
   | 10 % | 27.2 | 0 | 4 850 |

   The short-context planner is decode-bound (CPU launch) and loses little: 310 ms unlimited, 446 ms at 10 %.
5. **Bandwidth interference under an SM partition is small on this GPU.** Executor alone vs next to a continuous planner
   on the remaining SMs: 90 % SMs 25.5 → 26.3–26.6 ms (+3–4 %), 85 % 26.2 → 27.4–27.6 ms (+5 %), 75 % 27.5 → 29.0–29.2 ms
   (+5–6 %), 50 % 31.8 → 35.2–36.5 ms (+11–15 %). DRAM utilization (smi) rises from 27 % alone to 38–58 % co-located.
6. **Memory, not compute, caps co-location on 32 GB.** Two executor processes (15.3 GB each without text encoders) is the
   maximum; an executor with its text encoder plus a 16 k-context planner already exceeds the card.

The 5090 has ~1.8 TB/s DRAM; whether observation 5 holds on Jetson AGX Thor (shared LPDDR5X, ~273 GB/s, CPU + GPU on one
memory) is the open question for the Thor run.

## R3: sessions and executor processes

One frontend serving several sessions (`session_switch.py`, `r3/session_switch.json`):

| item | value |
|---|---|
| per-session state | text context 513 × 7680 BF16 = 7.9 MB |
| `set_prompt`, new instruction (live Qwen3-4B encode) | 120.7 ms |
| `set_prompt`, cached context (copy) | 0.05 ms |
| `infer()` | 24.3 ms |
| weights / peak allocated | 23.2 / 23.4 GB, independent of session count |

| sessions round-robin | tick p50 ms | aggregate Hz | per-session Hz |
|---|---|---|---|
| 1 | 24.3 | 41.1 | 41.1 |
| 2 | 24.5 | 40.9 | 20.4 |
| 4 | 24.7 | 40.1 | 10.0 |
| 8 | 24.7 | 40.1 | 5.0 |

Several executor processes (`run_r3.sh`): one process alone runs at p50 24.9 / p99 25.2 ms. A second executor process
does not fit, under time-slicing or MPS, even with loads staggered by 60 s: the second process OOMs while loading, since
loading needs the text encoder resident (~23 GB peak) next to the first process's 15.3 GB. Sessions on one 32 GB card
are therefore served by one frontend, not by one process each. (smi means in `r3/*.json` include the idle wait before
`START_AT` and are not comparable to R1.)

## R4: kernels in the planner's long-context prefill

`planner_prefill_kernels.py`, torch profiler (CUDA activity), one model forward over 16 261 prompt tokens, SDPA attention;
`r4/prefill_kernels_ctx16000.json` (+ chrome trace).

| item | value |
|---|---|
| kernels / total kernel time / span | 2 434 / 536.7 ms / 542.6 ms |
| longest kernel | 42.9 ms, CUTLASS BF16 GEMM grid 1024 × 149: the LM head over all 16 k positions (a plain forward; `generate` normally computes logits for the last position only) |
| attention (FlashAttention-2 fwd) | 28 calls, 5.58 ms each, grid 128 × 16 (2 048 blocks), 155.7 ms total |
| projection / MLP GEMMs (nvjet sm120) | 196 calls, ≤ 2.2 ms each, 220 ms total |

No single kernel approaches the ~290 ms executor stall seen under `prio` / `mps100`.

Executor and planner on one timeline (`overlap_trace.py`, in-process `prio` setup, one profiler, `r4/overlap_ctx16000.json`):
one long-context planner call (traced prefill 713 ms, decode 412 ms) with executor ticks back-to-back beside it. An
executor tick is K = 5 240 kernels (stage_inputs + graph replay). wait = span − executor kernel time; planner cover =
fraction of the tick span during which a planner kernel runs.

| executor tick | span ms | executor kernel time ms | wait ms | longest gap ms | planner cover |
|---|---|---|---|---|---|
| alone (5 ticks) | 25.8–26.2 | 22.1 | 3.7–4.0 | — | 0 |
| before the planner's GPU work | 26.3 | 22.1 | 4.2 | 0.30 | 0.01 |
| prefill begins | 121.3 | 28.5 | 92.8 | 4.73 | 0.86 |
| inside prefill | 307.1 | 34.7 | 272.4 | 4.75 | 0.99 |
| prefill ends | 119.5 | 31.9 | 87.6 | 4.74 | 0.91 |
| decode (10 ticks) | 32.2–37.5 | 25.6–28.7 | 6.6–9.0 | 0.38–0.40 | 0.45–0.70 |

The executor's own kernels run 1.6× slower inside the prefill (shared SMs and bandwidth), but 89 % of the stalled tick
is waiting at kernel boundaries: averaged over 5 240 kernels, ~52 µs per kernel, bounded by the planner's kernel
lengths (longest gap 4.75 ms ≈ one attention kernel of 5.6 ms). Stream priority acts when SM slots free up, so it does
not shorten these waits. An SM partition removes them, and so would fewer, larger executor kernels.

## R5: SM partition with green contexts (no MPS)

`green_partition.py` splits the SMs into two green contexts inside one process; `colocation_bench.py green<P>` runs the
planner thread on P % of the SMs and the executor on the rest; `run_r5.sh`, raw results in `r5/`. A CUDA graph runs on
the SMs of the stream it was **captured** on: replaying the frontend's original graph on a green-context stream still
uses all SMs (executor on 42 of 170 SMs: 24.7 ms, unchanged), so the bench recaptures the graph on the executor's
partition (then 50.4 ms on 42 SMs). Eager kernels follow the launch stream (8192³ BF16 matmul: 18.8 ms on 40 SMs,
5.9 ms on 130 SMs).

| planner context | planner SMs (of 170) | exec p50 ms | exec p99 ms | miss % | planner call ms | same MPS cell: exec p99 / planner call |
|---|---|---|---|---|---|---|
| — (executor alone on 122 / 85 SMs) | — | 28.7 / 32.4 | 29.0 / 32.7 | 0 | — | 27.8 / 32.1 (75 % / 50 %) |
| short | 48 (25 %) | 30.8 | 31.1 | 0 | 325 | 29.5 / 321 |
| short | 85 (50 %) | 36.8 | 37.2 | 0 | 307 | 35.5 / 303 |
| long | 48 (25 %) | 30.4 | 34.2 | 0 | 1 820 | 31.5 / 2 065 |
| long | 85 (50 %) | 37.5 | 45.0 | 0 | 1 150 | 42.1 / 1 202 |

Green contexts give the same protection as the MPS SM limit within ~3 ms of executor p99, in one process and without
a daemon (25 % requested → 48 SMs: the split rounds up to the device's SM granularity).

## Jetson AGX Thor (round 1028)

Same scripts on the Thor (`run_thor.sh`; 20 SMs, MAXN, GPC locked, EMC not locked; transformers 5.17.0 from an isolated
directory). Executor alone p50 / p99 163 / 170 ms, so the control period is 260 ms (1.5 × p99). Full table and per-cell
planner calls in `THOR_STATUS_SUMMARY.md` round `1028`.

| | Thor | RTX 5090 |
|---|---|---|
| planner call, 259 / 16k / 32k tokens | 4.9 / 7.7 / 12.1 s | 0.30 / 0.79 / 1.55 s |
| green-context split | 2, 4, 8 SMs requested → 8 | 43 requested → 48 |
| executor alone on 12 / 4 SMs (inferred split) vs 20 SMs, p50 | 157 / 234 vs 163 ms | — |

Executor p99 ms and misses at the 260 ms period (the planner runs continuously; `mps25` crashed in cuDNN in both
contexts):

| mode | short context | long context (16k) |
|---|---|---|
| serial | 5 112, 100 % | 7 866, 100 % |
| process | 390–393, 100 % | 393–414, 97–98 % |
| prio | 205–210, 0 % | 1 284–1 327, 2.4 % |
| green25 (planner 8 SMs) | 198, 0 % | 2 380, 3.0 % |
| green50 | 334, 11.6 % | 1 596, 26.9 % |
| mps50 | 306, 12.3 % | 225, 0 % |
| mps100 | 225, 0 % | 1 714, 5.0 % |

- The isolated executor's p50 moved from 162 to 126 ms between the start and the end of the grid (p99 163 → 165), so
  p50 differences in this round are not effects; misses and second-scale p99 values are.
- Long context breaks the in-process modes on the Thor: `green25` stalls up to 2.4 s although the same split protects
  the executor with a short context and on the 5090 (p99 34 ms). Only the process-level MPS limit at 50 % is clean.
  Cause open (issues.md ISSUE-098).
- With SMs split, sharing costs the executor 1–3 % at p50 and 21–41 % at p99 (5090: 3–15 % at p99).
- The torch profiler records no kernel events on the Thor and `overlap_trace.py` segfaults in the graph launch
  (issues.md ISSUE-099), so the R4 kernel-level timeline has no Thor counterpart yet.

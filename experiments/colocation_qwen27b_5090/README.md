# Qwen3.8-27B planner + ImageWAM executor on one RTX 5090 (FlashRT)

Question: when a 27B VLM planner and a real-time WAM executor share one 32 GB RTX 5090, both on FlashRT, where does
the executor lose its deadline, through which resource (SMs, DRAM bandwidth, the host launch path, memory capacity,
power), and what does each protection mechanism cost the planner?

## Setup

| item | value |
|---|---|
| GPU | RTX 5090 32 GB (31.36 GiB usable), driver 595.91.07, clocks unmanaged (2.6-2.9 GHz observed), power cap 575 W; no other user's process during a cell (checked every 2 s, `foreign_gpu_pids`) |
| executor | ImageWAM LIBERO (FLUX.2 klein 4B + ActionDiT), FlashRT fp8, 10 denoise steps, 2 views 224 x 224; one tick = `stage_inputs` + CUDA-graph replay (VAE, prefill, denoise); graph of 5 240 kernels |
| executor schedule | period 50 ms (20 Hz); latency = finish - release; an overrun delays the next release; miss = latency > 50 ms |
| executor memory | slimmed after `set_prompt` (`exec_slim.py`): Qwen3-4B text encoder and the FP16 sources of the 140 quantized linears freed, 22.96 -> 6.30 GB allocated; actions bit-identical after slimming and after the freed memory is refilled with NaN (`cells/exec_slim.json`) |
| planner | Qwen3.8-27B distilled checkpoint, FlashRT NVFP4 (`Qwen36VlTorchFrontendRtx`), FP8 KV, chunked prefill (2048 rows), MTP speculative decode, thinking off; prompt = front + wrist LIBERO frame (2 x 256 image tokens) + planning instruction (JSON answer), 64 new tokens per call |
| planner context | short: 602 tokens; long: 8 110 tokens (the same prompt after a text execution history) |
| planner rate | back-to-back calls (worst case) |
| window | 60 s per cell after warm-up (10 executor ticks, 3 planner calls) |
| telemetry | nvidia-smi every 100 ms (utilization, DRAM utilization, SM / memory clock, power), timestamped; per tick: event span, replay span, host split (stage_inputs / graph launch / wait) |

Sharing modes (`colocation_bench.py`):

| mode | mechanism |
|---|---|
| `serial` | one thread; a planner call runs to completion between two executor ticks |
| `prio` | planner thread in the executor's process; executor stream priority -1 |
| `green<P>` | one process, two green contexts: planner thread on P % of the SMs (25 % -> 48 SMs, 50 % -> 88 SMs), executor on the rest |
| `process` | planner in a second process; the driver time-slices the two contexts |
| `mps<P>` | planner process under MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` P, executor process 100 - P (`mps100`: no limit) |
| `coop` | as `prio`, plus `coop_gate.py`: the planner starts each GPU segment (ViT, 256-token prefill chunk, decode cycle) only when the executor is idle and the segment fits before the next release, or at the start of a slack window when it never fits |
| `greenhog<P>` | `bandwidth_hog.py` (back-to-back 256 MB device copies) on P % of the SMs in place of the planner |
| `greenalone<P>`, `mps_exec<E>_alone` | executor alone on the same SM share (denominators) |

Files: `exec_slim.py`, `planner_qwen27b.py`, `colocation_bench.py`, `coop_gate.py`, `bandwidth_budget.py`,
`stream_check.py`, `exec_capture_check.py`, `summarize.py`, `report_tables.py`, `run_grid.sh`, `run_followup*.sh`, `run_gil_profile.sh`;
raw cells in `cells/`, tables in `cells_summary.md`.

## Planner alone (`cells/planner_alone_sm*.json`, medians of 5 calls)

| planner SMs | ViT ms | prefill ms (short / long) | decode ms per spec cycle (short / long) | call ms (short / long) |
|---|---|---|---|---|
| 170 (whole device) | 13.4 | 54 / 668 | 23.8 / 27.9 | 666 / 1 334 |
| 122 | 16.8 | 60 / 773 | 23.8 / 28.0 | 676 / 1 446 |
| 88 | 20.8 | 80 / 1 036 | 25.1 / 29.3 | 732 / 1 741 |
| 48 | 32.4 | 126 / 1 725 | 27.5 / 32.5 | 851 / 2 516 |
| 170, 256-token prefill chunks | 13.4 | 78 / 959 | 23.8 / 25.3 | 715 / 1 692 |
| 170, `FLASHRT_QWEN36_MAX_Q_SEQ=1024` | 13.4 | 54 / 699 | 23.9 / 25.4 | 667 / 1 384 |

- 23-26 spec cycles per call (2.5-2.8 tokens per cycle); the output is the same JSON plan in every configuration
  except the 256-chunk one (a different but valid plan).
- Peak memory 22.75 GB (20.74 GB with `MAX_Q_SEQ` 1024).
- `bandwidth_budget.py` (`bandwidth_budget.json`): one verify forward reads 14.45 GB of weights, one MTP draft
  1.03 GB, so one cycle reads 17.5-20.6 GB (K = 3-6), about 0.74-0.87 TB/s at the measured cycle times (derived);
  a 1 GiB device copy reaches 1.52 TB/s.

## Co-location (`report_tables.py` -> `report_tables.md`; per-phase and host-split tables in `cells_summary.md`)

Executor slowdown = p50 / p50 of the executor alone on the same SM share (whole device 28.7 ms, 122 SMs 30.9 ms,
82 SMs 35.1 ms, MPS 75 % 32.9 ms, MPS 50 % 38.5 ms). Planner slowdown = call / call of the planner alone on the whole
device with the same planner settings. `process` and `mps100` use `MAX_Q_SEQ` 1024 in the planner (with 2048 the
planner process runs out of memory at load); `coop` uses 256-token prefill chunks. Host-side variants of the
in-process modes (research-only, the frontend code is unchanged): "switch 0.5 ms" = `GIL_SWITCH_MS=0.5`; "readback
outside GIL" = `PLANNER_READBACK_NOGIL=1` (the per-cycle `cuda_read_i32_sync` first waits on the stream through torch,
which releases the GIL) with `CAPTURE_THREAD_LOCAL=1`; "prefill graphs" = `FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1`.
`short prio, switch 0.5 ms` crashed (see observation 3) and has no row.

### planner context: short

| mode | planner / executor SMs | exec p50 ms | exec p99 ms | miss % | exec ticks/s | exec slowdown (p50, same SMs alone) | planner call ms | planner slowdown | prefill ms | decode ms / cycle | power W | SM MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| serial | — | 695.9 | 708.9 | 100.0 | 1.4 | 24.25x | 667 | 1.00x | 54 | 23.8 | 430 | 2850 |
| process (time-slicing) | — | 64.0 | 65.6 | 95.0 | 15.9 | 2.23x | 1353 | 2.03x | 105 | 47.9 | 446 | 2834 |
| mps100 | 100 % / 100 % | 41.6 | 58.6 | 10.3 | 19.8 | 1.45x | 908 | 1.36x | 66 | 32.7 | 552 | 2823 |
| prio | — | 114.2 | 246.1 | 84.8 | 8.3 | 3.98x | 791 | 1.19x | 55 | 27.0 | 464 | 2839 |
| green25 | 48 / 122 | 140.2 | 265.0 | 80.4 | 7.5 | 4.54x | 951 | 1.43x | 131 | 30.1 | 423 | 2860 |
| green50 | 88 / 82 | 112.5 | 234.8 | 82.4 | 8.2 | 3.20x | 866 | 1.30x | 85 | 28.3 | 446 | 2851 |
| mps50 | 50 % / 50 % | 46.1 | 46.7 | 0.0 | 20.0 | 1.20x | 905 | 1.36x | 92 | 30.2 | 565 | 2834 |
| mps25 | 25 % / 75 % | 37.7 | 38.2 | 0.0 | 20.0 | 1.15x | 1261 | 1.89x | 159 | 40.6 | 511 | 2843 |
| coop | — | 37.1 | 52.4 | 8.5 | 19.9 | 1.29x | 1704 | 2.38x | 164 | 55.9 | 455 | 2831 |
| green25, switch 0.5 ms | 48 / 122 | 129.5 | 239.2 | 81.8 | 8.1 | 4.19x | 962 | 1.44x | 131 | 30.4 | 426 | 2859 |
| coop, switch 0.5 ms | — | 37.1 | 52.3 | 8.6 | 20.0 | 1.29x | 1686 | 2.36x | 165 | 55.8 | 457 | 2831 |
| prio, readback outside GIL | — | 36.9 | 46.6 | 0.0 | 20.0 | 1.29x | 950 | 1.43x | 68 | 33.5 | 531 | 2826 |
| green25, readback outside GIL | 48 / 122 | 35.5 | 53.9 | 2.3 | 20.0 | 1.15x | 1050 | 1.58x | 138 | 33.4 | 522 | 2832 |
| green25, readback outside GIL, prefill graphs | 48 / 122 | 35.5 | 36.1 | 0.1 | 20.0 | 1.15x | 1050 | 1.58x | 138 | 33.3 | 524 | 2832 |
| green50, readback outside GIL, prefill graphs | 88 / 82 | 41.2 | 45.4 | 0.0 | 20.0 | 1.17x | 1036 | 1.56x | 89 | 35.0 | 532 | 2836 |
| coop, readback outside GIL | — | 30.1 | 32.3 | 0.0 | 20.0 | 1.05x | 1500 | 2.10x | 161 | 50.3 | 474 | 2829 |

### planner context: long

| mode | planner / executor SMs | exec p50 ms | exec p99 ms | miss % | exec ticks/s | exec slowdown (p50, same SMs alone) | planner call ms | planner slowdown | prefill ms | decode ms / cycle | power W | SM MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| serial | — | 1362.9 | 1364.4 | 100.0 | 0.8 | 47.50x | 1334 | 1.00x | 671 | 27.8 | 500 | 2714 |
| process (time-slicing) | — | 64.9 | 66.3 | 97.1 | 15.5 | 2.26x | 2740 | 1.98x | 1368 | 50.8 | 493 | 2796 |
| mps100 | 100 % / 100 % | 42.8 | 96.3 | 33.8 | 15.7 | 1.49x | 1767 | 1.28x | 824 | 35.0 | 562 | 2688 |
| prio | — | 105.7 | 297.4 | 91.8 | 7.9 | 3.68x | 1544 | 1.16x | 770 | 30.8 | 516 | 2702 |
| green25 | 48 / 122 | 149.7 | 758.6 | 88.1 | 4.6 | 4.85x | 2618 | 1.96x | 1744 | 34.7 | 404 | 2860 |
| green50 | 88 / 82 | 183.9 | 451.7 | 87.1 | 5.7 | 5.24x | 1884 | 1.41x | 1058 | 32.5 | 494 | 2833 |
| mps50 | 50 % / 50 % | 46.1 | 47.0 | 0.0 | 20.0 | 1.20x | 2074 | 1.55x | 1236 | 32.2 | 570 | 2742 |
| mps25 | 25 % / 75 % | 35.8 | 38.0 | 0.0 | 20.0 | 1.09x | 3443 | 2.58x | 2203 | 42.4 | 520 | 2835 |
| coop | — | 35.5 | 56.0 | 7.4 | 19.9 | 1.24x | 3362 | 1.99x | 1634 | 58.8 | 511 | 2796 |
| prio, switch 0.5 ms | — | 68.3 | 234.9 | 94.3 | 10.7 | 2.38x | 1626 | 1.22x | 853 | 31.1 | 522 | 2668 |
| green25, switch 0.5 ms | 48 / 122 | 61.8 | 379.9 | 90.5 | 11.8 | 2.00x | 2708 | 2.03x | 1829 | 35.2 | 466 | 2840 |
| coop, switch 0.5 ms | — | 35.5 | 55.4 | 7.4 | 19.9 | 1.24x | 3359 | 1.98x | 1634 | 58.6 | 511 | 2797 |
| prio, readback outside GIL | — | 40.0 | 164.7 | 29.6 | 15.9 | 1.39x | 1744 | 1.31x | 815 | 38.2 | 555 | 2692 |
| green25, readback outside GIL | 48 / 122 | 33.1 | 516.1 | 16.3 | 12.1 | 1.07x | 2950 | 2.21x | 1768 | 48.5 | 456 | 2846 |
| prio, readback outside GIL, prefill graphs | — | 40.3 | 61.0 | 44.2 | 18.7 | 1.40x | 1876 | 1.41x | 938 | 38.2 | 554 | 2671 |
| green25, readback outside GIL, prefill graphs | 48 / 122 | 34.1 | 35.9 | 0.0 | 20.0 | 1.10x | 3050 | 2.29x | 1877 | 48.1 | 524 | 2822 |
| green50, readback outside GIL, prefill graphs | 88 / 82 | 42.0 | 44.6 | 0.1 | 20.0 | 1.20x | 2400 | 1.80x | 1202 | 49.3 | 537 | 2750 |
| coop, readback outside GIL | — | 33.2 | 38.0 | 0.0 | 20.0 | 1.16x | 3100 | 1.83x | 1634 | 49.6 | 526 | 2792 |

## Observations

1. **In one process the executor loses its deadline on the host, not on the GPU.** With green contexts the executor's
   graph runs 10-20 % slower than alone (35.1 vs 30.9 ms, long green25), but `stage_inputs` (0.5 ms alone) waits
   24-118 ms for the GIL (`cells_summary.md`, host split). py-spy (`run_gil_profile.sh`, `--gil --native`, prio short)
   puts 632 of the planner thread's 683 GIL-holding samples (93 %) in `fvk.cuda_read_i32_sync` at
   `qwen36_rtx.py:10077`: the per-cycle readback of the accepted draft count, a `cudaMemcpyAsync` +
   `cudaStreamSynchronize` that waits for the whole verify forward with the GIL held (the FlashRT kernel bindings do
   not release the GIL). Waiting outside the GIL first takes short prio from 84.8 % misses to 0 %.
2. **Long context has a second host-side source: the eager chunked prefill** (a Python loop launching kernels). A
   0.5 ms switch interval halves the long-context waits but leaves the decode-bound short context unchanged (55.9 ms
   `stage_inputs` p50), since the interval only applies between bytecodes. With the prefill as CUDA graphs on top of
   the readback change, long green25 goes from 88.1 % misses (p99 759 ms) to 0 % (p99 35.9 ms) with the planner at
   2.29x, against 2.58x for the two-process mps25.
3. **The planner captures graphs during the window** (in prio the planner's captured graphs grow from 46-64 to
   86-110 within 60 s). Under the default global capture mode a CUDA call from the executor thread during such a
   capture fails it: runs crashed with `cudaErrorStreamCaptureUnsupported` / `StreamCaptureInvalidated` whenever the
   thread interleaving changed (py-spy, 0.5 ms switch interval, readback outside the GIL); thread-local capture removes
   the crash.
4. **Without an SM limit the prefill takes the SMs.** prio with both host-side changes: long-context p99 61 ms but
   44.2 % misses, the executor graph taking 55-60 ms while a graph-captured prefill fills the GPU; mps100 likewise
   misses 33.8 % at long context. Limiting the planner's SMs meets the deadline (green25 / green50 with both changes,
   mps25, mps50: 0-0.1 %) and costs mainly prefill (2.4-3.3x at 48 SMs / 25 %), since decode hardly depends on SMs.
5. **Bandwidth is what remains after SM isolation.** Executor slowdown against the same SM share alone: 1.07-1.20x with
   the planner on disjoint SMs; 1.85x / 3.04x with back-to-back copies (1.34 / 1.44 TB/s) on 48 / 88 SMs.
6. **Time-slicing (`process`) halves both** (executor 64-65 ms, 95-97 % misses; planner 2.0x). **coop** with the
   readback change gives the lowest executor latency (1.05-1.16x, 0 % misses) and costs the planner 1.8-2.1x (one
   decode cycle per 50 ms period, 256-token chunks); without the change its residual misses (7-9 %) are the executor
   waking up behind a decode cycle that overlaps the next release (the cycle, 25-30 ms, is longer than the 21 ms slack).
7. **Power reaches the cap:** 520-570 W in the co-located cells against 314-321 W for the executor alone; mean SM clock
   2 598-2 872 MHz (2 841-2 855 alone).
8. **Memory:** in one process 28.7-29.0 GB allocated (2.6-2.9 GB left). Two processes need the planner's prefill
   scratch at 1024 rows (`MAX_Q_SEQ`, -2.0 GB, long prefill +5 %) for `process` / `mps100`; mps50 / mps25 started with
   0.01-0.14 GB left. The executor runs without its text encoder, so a new instruction from the planner cannot be
   encoded on the GPU without loading Qwen3-4B again.
9. Executor alone is 28.2-28.8 ms with the cuBLAS this environment loads (13.4.1.3), independent of where its graph is
   captured (`cells/exec_capture_check.json`) and of the kernel-library build (`cells/so_compare/`); with cuBLAS
   13.6.1.10 preloaded it is 24.4 ms (`issues.md` ISSUE-103). Every cell of this README ran with 13.4.1.3. The cells
   above construct both models on the default stream and then move them to their streams; side-stream construction
   corrupted the planner's weights until the loader fix (`issues.md` ISSUE-104).

## Host-side fixes in FlashRT (`run_hostfix.sh` -> `cells/hostfix/`)

The monkeypatches above are now changes in FlashRT itself, and these cells run with both `PLANNER_READBACK_NOGIL` and
`CAPTURE_THREAD_LOCAL` off:

- `flash_rt_kernels.cuda_read_i32_sync` waits with the GIL released, draining the stream before it enqueues the copy.
  Copying first (`cells/hostfix/short_prio_p1a_copy_then_sync.json`) left staging at 0.6 ms but the executor's graph
  launch at 37.5 ms p50 (14 ms alone) and 92.4 % misses: a device-to-pageable-host copy enqueued behind pending work
  waits inside the driver and stalls the other thread's launches.
- Every graph capture of `qwen36_rtx.py`, `_qwen3_vl_vision_rtx.py` and `imagewam_thor.py` uses thread-local capture
  mode: no capture failure in 21 windows, one of them under py-spy sampling.
- The Qwen3.6 NVFP4 loader runs its conversion kernels on the current stream (ISSUE-104 resolved): a side-stream
  construction gives the same 1 482 weight buffers and outputs as a default-stream one
  (`smoke/stream_check_{plain,side}_fix.json`; before the fix 48 buffers differed and the output was garbage).

Executor p50 / p99 ms and miss rate, planner call ms, against the monkeypatched cells (prefill graph on, except coop:
eager 256-token chunks):

| cell | FlashRT fixes | monkeypatches | planner call ms (fixes / monkeypatches) |
|---|---|---|---|
| prio, short | 37.3 / 47.9 / 0.1 % | 36.9 / 46.6 / 0 % | 950 / 950 |
| green25, short | 35.5 / 36.2 / 0.1 % | 35.5 / 36.1 / 0.1 % | 1 050 / 1 050 |
| green50, short | 41.3 / 45.5 / 0 % | 41.2 / 45.4 / 0 % | 1 009 / 1 036 |
| green25, long | 34.2 / 36.0 / 0 % | 34.1 / 35.9 / 0 % | 3 050 / 3 050 |
| green50, long | 42.1 / 45.1 / 0.1 % | 42.0 / 44.6 / 0.1 % | 2 400 / 2 400 |
| coop, short | 30.3 / 32.7 / 0 % | 30.1 / 32.3 / 0 % | 1 550 / 1 500 |
| coop, long | 33.4 / 38.2 / 0 % | 33.2 / 38.0 / 0 % | 3 100 / 3 100 |
| prio, long (no SM limit) | 40.5 / 61.1 / 44.4 % | 40.3 / 61.0 / 44.2 % | 1 876 / 1 876 |

- The single missed ticks of the 0.1 % cells fall in the planner's first call of the window.
- Short-context green25 needs the prefill graph too: without it 2.4-3.6 % misses (p99 53-56 ms, three windows), as the
  monkeypatched cell without it (2.3 %).
- Output checks (`EXEC_CHECK_ACTIONS=1`: fixed executor noise, per-tick action diff, per-call planner graph counts):
  the planner alone gives the same output CRC as before the changes. Under co-location, a planner call that captures
  graphs during the window can give a different output (1-2 of about 60 calls in `prio` short; ISSUE-105), and a few
  executor ticks differ by up to 4.9e-3 where the two models share SMs (`prio`, `coop`; 0 of about 1 200 per `green`
  window; ISSUE-106).


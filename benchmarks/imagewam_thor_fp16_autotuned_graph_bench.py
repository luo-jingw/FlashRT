#!/usr/bin/env python
"""KNOWN TO HANG -- do not run without fixing the issue below first.
See opportunities.md OPT-004 "Suggested Next Step -- attempted, hung,
shelved" for the full record.

OPT-004: does combining CUDA Graph capture WITH pre-autotuned GEMMs
give an additive win over either alone?

Reuses FullImageWAMFP16Autotuned (benchmarks/imagewam_thor_fp16_autotuned_bench.py)
unmodified: run it uncaptured a few times first (on a side stream) so
every _Fp16Linear's lazy, one-time `autotune_fp16_nn` call actually
fires during warmup -- by the time capture starts, every GEMM call
inside the captured region is a plain, graph-safe `fp16_nn` already
pointed at its autotuned cuBLASLt algorithm. Then capture ONE more
full run (prefill + 10-step denoise) into a CUDAGraph and time replay.

**As actually run on this machine: this hangs.** 66+ minutes of
CPU/GPU time with zero new output, killed rather than left running
further. Suspected, not confirmed, cause: `autotune_cached`'s own C++
implementation hardcodes stream 0 for its internal benchmark loop
(`autotune_fp16_nn` doesn't even accept a stream argument), while this
script's warmup runs on an explicit non-default side stream (needed so
the same stream can be used for graph capture afterward) -- combining
the two may deadlock in `cudaEventSynchronize`/`cudaDeviceSynchronize`.
Not root-caused further; shelved rather than debugged, since
autotune-alone (+4%/+10%) and graph-alone (+2%/+7.5%) are both already
confirmed, independently useful wins on their own. If revisiting: try
autotuning on the DEFAULT stream first (a separate warmup pass with no
`torch.cuda.stream(...)` context at all), and only switch to the side
stream for the capture call itself, never for anything that triggers
`autotune_fp16_nn`.

Compare against (same real dims, this machine, all post-OPT-003):
  graph-free, no autotune  (imagewam_thor_fp16_bench.py):            203.2 ms
  autotune only, no graph  (imagewam_thor_fp16_autotuned_bench.py):  195.1 ms
  graph only, no autotune  (imagewam_thor_graph_bench.py, real ImageWAMTorchFrontendThor): 198.5 ms
  graph + autotune (this script): UNKNOWN -- hangs, see warning above
"""
from __future__ import annotations

import statistics

import torch

# Reuse the already-built, already-verified autotuned pipeline class directly.
from imagewam_thor_fp16_autotuned_bench import FullImageWAMFP16Autotuned

WARMUP, ITERS = 15, 50
NUM_DENOISE_STEPS = 10


def main():
    print("Building autotuned FP16 pipeline at real dims "
          "(quantizes/allocates + autotunes every weight -- may take a while)...")

    model = FullImageWAMFP16Autotuned()
    torch.cuda.synchronize()
    print("Built (autotune log above). Warming up on a side stream to "
          "trigger every _Fp16Linear's one-time autotune call...")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.run_full(NUM_DENOISE_STEPS, s.cuda_stream)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print("Warmup done, every GEMM should now be autotuned. Capturing CUDA Graph...")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        model.run_full(NUM_DENOISE_STEPS, s.cuda_stream)
    torch.cuda.synchronize()
    print("Captured. Running steady-state replay timing "
          f"({WARMUP} warmup + {ITERS} measured replays)...\n")

    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p50, p90, mean = times[len(times) // 2], times[int(len(times) * 0.9)], statistics.mean(times)

    print(f"graph + autotune, full (prefill + {NUM_DENOISE_STEPS}-step denoise):")
    print(f"  P50={p50:8.3f} ms  P90={p90:8.3f} ms  mean={mean:8.3f} ms")
    print("\nCompare (this machine, all post-OPT-003):")
    print("  graph-free, no autotune : 203.2 ms")
    print("  autotune only, no graph : 195.1 ms")
    print("  graph only, no autotune : 198.5 ms")
    print(f"  graph + autotune (this) : {p50:.1f} ms")


if __name__ == "__main__":
    main()

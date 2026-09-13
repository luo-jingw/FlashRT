#!/usr/bin/env python
"""OPT-004 step 1: measure the CUDA-graph-captured whole-pipeline
latency (via the real ImageWAMTorchFrontendThor) at ImageWAM's real
dims, for direct comparison against the graph-free numbers already
measured in benchmarks/imagewam_thor_fp16_bench.py.

This answers the question OPT-004 posed: how much of the graph-free
"prefill + 10-step denoise" number is CPU/launch overhead that CUDA
Graph capture already eliminates, versus real GPU compute time that
only kernel-level work (fusion, better GEMM algorithms) could reduce
further. Cheap to answer -- no new kernel code, just exercising the
graph-capture path Phase 5 already built but never benchmarked at real
scale (every prior "full pipeline" number in this project was
deliberately graph-free, to isolate per-layer cost in isolation).

Same real dims as imagewam_thor_fp16_bench.py, same precision (FP16,
plain fp16_nn -- ImageWAMTorchFrontendThor has no quantization wired
in, see OPT-001), same num_denoise_steps=10, includes the OPT-003 fix
(ImageWAMAttnBackend.run()'s "mot_joint" branch already restricts Q to
action rows -- this frontend is unmodified code, so it picks the fix
up automatically).
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=128, a0=896, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=960,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / 10, num_denoise_steps=10,
)

WARMUP, ITERS = 15, 50


def main():
    print("Building ImageWAMTorchFrontendThor at real dims "
          "(quantizes/allocates every weight -- may take a while)...")
    print(f"Dims: {REAL_DIMS}\n")

    frontend = ImageWAMTorchFrontendThor(dims_override=REAL_DIMS)
    frontend.set_prompt("bench")  # triggers the one-time CUDA Graph capture
    torch.cuda.synchronize()
    print("Graph captured. Running steady-state infer() timing "
          f"({WARMUP} warmup + {ITERS} measured replays)...\n")

    obs = {"image": np.zeros((4, 4, 3), dtype=np.uint8)}

    for _ in range(WARMUP):
        frontend.infer(obs)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        frontend.infer(obs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p50, p90, mean = times[len(times) // 2], times[int(len(times) * 0.9)], statistics.mean(times)

    print(f"graph_captured full (prefill + 10-step denoise):")
    print(f"  P50={p50:8.3f} ms  P90={p90:8.3f} ms  mean={mean:8.3f} ms")
    print(f"\nCompare against benchmarks/imagewam_thor_fp16_bench.py's graph-free "
          f"'full (prefill + 10-step denoise)' number at the same dims (post-OPT-003 "
          f"fix: 203.2ms P50 on this machine) -- the gap between that number and "
          f"this one is Python/launch overhead CUDA Graph capture already removes; "
          f"anything left is real GPU compute time only kernel-level work can reduce.")


if __name__ == "__main__":
    main()

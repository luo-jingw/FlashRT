#!/usr/bin/env python
"""ImageWAM NVFP4-vs-FP8 GEMM-only M-sweep on RTX 5090 (SM120).

opportunities.md OPT-033's own GEMM-level comparison (nvfp4 15.59 ms/infer
vs fp8 12.36 ms/infer, ~80% of the 4.10 ms e2e gap) is a SUM over every
NVFP4 GEMM call inside one real `infer()` -- i.e. already a mix of
whatever M values ImageWAM's own layers happen to run at (large-M backbone
prefill, small-M ActionDiT denoise), not a controlled per-M measurement.
This script isolates M as its own axis: same real (N, K) shapes ImageWAM
actually uses (the same table `benchmarks/imagewam_thor_small_m_tile_sweep.py`
uses on Thor), M swept independently, GEMM-only device time isolated via
`torch.profiler` the same way OPT-033's own number was produced -- so this
answers whether the NVFP4-slower-than-FP8 gap holds at every M or crosses
over somewhere, not a different question than OPT-033 already answered.

Does NOT sweep GEMM variants (`FLASHRT_FP4_GEMM`): OPT-033 already swept
`plain`/`widen`/`pingpong` at one shape and found the default (`pingpong`)
within 0.2 ms of the best. This script keeps that default fixed and varies
only M, so a variant sweep is not repeated here -- if a given M's result is
surprising, re-check it against the other two variants by hand
(`FLASHRT_FP4_GEMM=plain`/`widen`) before trusting it.

Real (N, K) shapes (ImageWAM backbone/ActionDiT, same table as the Thor
sweep):

  shape                               N      K
  double_qkv                       9216   1024
  double_proj                      1024   3072
  double_mlp0                      8192   1024
  double_mlp2                      1024   4096
  single_linear1                  17408   1024
  single_linear2                   1024   7168

Usage:
  python benchmarks/imagewam_5090_nvfp4_m_sweep.py
  python benchmarks/imagewam_5090_nvfp4_m_sweep.py --shapes double_qkv,single_linear1
  python benchmarks/imagewam_5090_nvfp4_m_sweep.py --m-values 16,64,256,905 --iters 50

Requires `flash_rt.hardware.detect_arch() == "rtx_sm120"` (raises
otherwise -- this script is RTX 5090/SM120-specific, not portable to
Thor). Random weights; this is a speed-only microbenchmark, not a
correctness check (OPT-033 already established cosine is fine for both
precisions at this GEMM family).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass

import torch

from flash_rt.hardware import detect_arch
from flash_rt.models.imagewam.quant_linear import Nvfp4LinearSm120, StaticFp8Linear

SHAPES = {
    "double_qkv": (9216, 1024),
    "double_proj": (1024, 3072),
    "double_mlp0": (8192, 1024),
    "double_mlp2": (1024, 4096),
    "single_linear1": (17408, 1024),
    "single_linear2": (1024, 7168),
}
DEFAULT_M_VALUES = (16, 32, 64, 128, 256, 512, 905)

NVFP4_KERNEL_NAME = "fp4_w4a16_gemm_sm120_bf16out"
FP8_KERNEL_NAME = "nvjet_sm120"  # cuBLASLt-generated FP8 MMA, OPT-033's own identification


@dataclass
class Row:
    shape: str
    n: int
    k: int
    m: int
    precision: str
    gemm_only_us: float
    whole_call_us: float


def _gemm_only_us(fn, iters: int, kernel_substr: str) -> float:
    """Sum this call's device time for kernels whose name contains
    `kernel_substr`, over `iters` profiled calls, per call -- same
    technique OPT-033's own number was produced with."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for e in prof.key_averages():
        if kernel_substr in e.key:
            total_us += e.self_device_time_total
    return total_us / iters


def _whole_call_us(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us


def _bench_one(n: int, k: int, m: int, iters: int, warmup: int) -> tuple[Row, Row]:
    w = torch.randn(k, n, dtype=torch.float16, device="cuda").contiguous()
    x = torch.randn(m, k, dtype=torch.float16, device="cuda").contiguous()
    out_fp4 = torch.empty(m, n, dtype=torch.float16, device="cuda")
    out_fp8 = torch.empty(m, n, dtype=torch.float16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    fp4 = Nvfp4LinearSm120(w.data_ptr(), n, k)
    fp8 = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)

    def call_fp4():
        fp4(x.data_ptr(), out_fp4.data_ptr(), m, stream)

    def call_fp8():
        fp8(x.data_ptr(), out_fp8.data_ptr(), m, stream)

    for _ in range(warmup):
        call_fp4()
        call_fp8()
    torch.cuda.synchronize()

    fp4_gemm = _gemm_only_us(call_fp4, iters, NVFP4_KERNEL_NAME)
    fp8_gemm = _gemm_only_us(call_fp8, iters, FP8_KERNEL_NAME)
    fp4_whole = _whole_call_us(call_fp4, iters)
    fp8_whole = _whole_call_us(call_fp8, iters)
    return (
        Row("", n, k, m, "nvfp4", fp4_gemm, fp4_whole),
        Row("", n, k, m, "fp8_cublaslt", fp8_gemm, fp8_whole),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", default=",".join(SHAPES), help="comma list of shape names")
    p.add_argument("--m-values", default=",".join(str(m) for m in DEFAULT_M_VALUES))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", help="write rows as JSON to this path")
    args = p.parse_args()

    arch = detect_arch()
    if arch != "rtx_sm120":
        raise RuntimeError(f"this script is RTX 5090/SM120-specific; detect_arch()={arch!r}")

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    m_values = [int(m.strip()) for m in args.m_values.split(",") if m.strip()]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=".").stdout.strip()
    gpu = torch.cuda.get_device_name(0)

    print(f"# commit={commit} gpu={gpu!r} arch={arch}")
    print(f"{'shape':<16}{'M':>6}{'N':>8}{'K':>8}{'nvfp4 gemm us':>16}{'fp8 gemm us':>14}"
          f"{'gemm ratio':>12}{'nvfp4 call us':>16}{'fp8 call us':>14}{'call ratio':>12}")

    rows: list[Row] = []
    for shape in shapes:
        n, k = SHAPES[shape]
        for m in m_values:
            r4, r8 = _bench_one(n, k, m, args.iters, args.warmup)
            r4.shape = r8.shape = shape
            rows.extend([r4, r8])
            gemm_ratio = r4.gemm_only_us / r8.gemm_only_us
            call_ratio = r4.whole_call_us / r8.whole_call_us
            print(f"{shape:<16}{m:>6}{n:>8}{k:>8}{r4.gemm_only_us:>16.2f}{r8.gemm_only_us:>14.2f}"
                  f"{gemm_ratio:>12.3f}{r4.whole_call_us:>16.2f}{r8.whole_call_us:>14.2f}{call_ratio:>12.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"commit": commit, "gpu": gpu, "rows": [vars(r) for r in rows]}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""GEMM-only precision comparison at ImageWAM's real projection shapes,
runnable on THIS dev machine (Ada sm_89).

Not NVFP4 (Blackwell-only, see imagewam_thor_fp4_bench.py's own docstring)
-- this uses the SEPARATE, older SM80-family CUTLASS INT8/INT4 rowwise
GEMM path (csrc/gemm/cutlass_sm80_int4_rowwise.cu), built for Jetson Orin
SM87's QuaRot path but templated on cutlass::arch::Sm80, which Ada's own
tensor cores support directly -- confirmed by actually building and
running it here (FLASHRT_ENABLE_CHAMELEON=ON, ENABLE_SM80_INT8_CUTLASS=ON,
GPU_ARCH=89; see PROJECT.md for the reconfigure command).

Correctness NOT validated (random weights, random already-packed INT4/INT8
bytes -- this GEMM's real numerical scheme needs a QuaRot Hadamard
rotation on both activation and weight before quantizing to survive int4's
dynamic range, csrc/kernels/fht_int4.cu; this script skips it entirely
since only GEMM throughput is being measured). This is a raw isolated-GEMM
comparison, not the full ImageWAM pipeline -- no attention, no norm, no
residual, no per-layer loop.
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
WARMUP, ITERS = 20, 100

# Real ImageWAM backbone GEMM shapes (this pipeline's own reduced-K/V
# convention, see plan.md Phase 3/OPT-002) at a representative M
# (a0=896, the backbone prefill sequence length).
SHAPES = {
    "q/proj [M=896,N=3072,K=3072]": (896, 3072, 3072),
    "k/v    [M=896,N=128, K=3072]": (896, 128, 3072),
    "mlp0   [M=896,N=9216,K=3072]": (896, 9216, 3072),
    "mlp2   [M=896,N=3072,K=9216]": (896, 3072, 9216),
}


def _time_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], statistics.mean(times)


def bench_fp16(m, n, k):
    gemm = fvk.GemmRunner()
    a = torch.randn(m, k, dtype=torch.float16, device=DEV)
    b = torch.randn(k, n, dtype=torch.float16, device=DEV)  # (K,N), fp16_nn convention
    d = torch.zeros(m, n, dtype=torch.float16, device=DEV)
    return _time_ms(lambda: gemm.fp16_nn(a.data_ptr(), b.data_ptr(), d.data_ptr(), m, n, k, 0))


def bench_fp8(m, n, k):
    a16 = torch.randn(m, k, dtype=torch.float16, device=DEV)
    w16 = torch.randn(k, n, dtype=torch.float16, device=DEV)  # (K,N) convention, same as fp16_nn
    a_f8 = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEV)
    w_f8 = torch.empty(k, n, dtype=torch.float8_e4m3fn, device=DEV)
    a_scale = torch.tensor([1.0 / 448.0], dtype=torch.float32, device=DEV)
    w_scale = torch.tensor([1.0 / 448.0], dtype=torch.float32, device=DEV)
    fvk.quantize_fp8_static_fp16(w16.data_ptr(), w_f8.data_ptr(), w_scale.data_ptr(), k * n, 0)
    d = torch.zeros(m, n, dtype=torch.float16, device=DEV)

    def run():
        fvk.quantize_fp8_static_fp16(a16.data_ptr(), a_f8.data_ptr(), a_scale.data_ptr(), m * k, 0)
        fvk.fp8_gemm_descale_fp16(a_f8.data_ptr(), w_f8.data_ptr(), d.data_ptr(),
                                   m, n, k, a_scale.data_ptr(), w_scale.data_ptr(), 0)

    return _time_ms(run)


def bench_int4(m, n, k):
    assert k % 32 == 0, f"K={k} must be 32-aligned for this SM80 INT4 kernel"
    # Random already-packed s4 bytes -- content is irrelevant for a pure
    # throughput measurement (see module docstring: no QuaRot rotation,
    # no correctness claim).
    a_packed = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=DEV)
    b_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEV)
    act_scale = torch.ones(m, dtype=torch.float32, device=DEV)
    weight_scale = torch.ones(n, dtype=torch.float32, device=DEV)
    d = torch.zeros(m, n, dtype=torch.float16, device=DEV)

    def run():
        rc = fvk.cutlass_int4_rowwise_fp16out(
            a_packed.data_ptr(), b_packed.data_ptr(),
            act_scale.data_ptr(), weight_scale.data_ptr(),
            d.data_ptr(), m, n, k, 0)
        if rc != 0:
            raise RuntimeError(f"cutlass_int4_rowwise_fp16out failed rc={rc}")

    return _time_ms(run)


def bench_int8(m, n, k):
    a_packed = torch.randint(-127, 127, (m, k), dtype=torch.int8, device=DEV)
    b_packed = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=DEV)
    act_scale = torch.ones(m, dtype=torch.float32, device=DEV)
    weight_scale = torch.ones(n, dtype=torch.float32, device=DEV)
    d = torch.zeros(m, n, dtype=torch.float16, device=DEV)

    def run():
        rc = fvk.cutlass_int8_rowwise_fp16out(
            a_packed.data_ptr(), b_packed.data_ptr(),
            act_scale.data_ptr(), weight_scale.data_ptr(),
            d.data_ptr(), m, n, k, 0)
        if rc != 0:
            raise RuntimeError(f"cutlass_int8_rowwise_fp16out failed rc={rc}")

    return _time_ms(run)


def _try(fn, *args):
    try:
        p50, _ = fn(*args)
        return f"{p50:9.3f}ms"
    except Exception as e:  # noqa: BLE001 -- report and keep going, not fatal
        return f"FAIL({type(e).__name__})"


def main():
    print(f"Ada (sm_89) GEMM-only precision comparison, warmup={WARMUP} iters={ITERS}\n")
    fp8_probe = _try(bench_fp8, 64, 64, 64)
    if fp8_probe.startswith("FAIL"):
        print(f"NOTE: cuBLASLt FP8 (E4M3) matmul heuristic search returns "
              f"CUBLAS_STATUS_NOT_SUPPORTED on this environment (confirmed at "
              f"a trivial 64x64x64 shape via two independent code paths -- "
              f"GemmRunner.fp8_nn_dev_fp16 and the standalone "
              f"fp8_gemm_descale_fp16 -- both fail identically). torch=2.10.0+cu128, "
              f"cuBLASLt=12.8.04, compute capability (8,9) -- a real, confirmed "
              f"environment limitation of THIS venv, not a shape or logic bug; "
              f"the user's own Thor run got real FP8 numbers, so this is Ada/venv-"
              f"specific. FP8 column will read FAIL below.\n")

    print(f"{'shape':35s} {'fp16':>12s} {'fp8':>18s} {'int8(SM80)':>14s} {'int4(SM80)':>14s}")
    for label, (m, n, k) in SHAPES.items():
        r16 = _try(bench_fp16, m, n, k)
        r8 = _try(bench_fp8, m, n, k)
        ri8 = _try(bench_int8, m, n, k)
        ri4 = _try(bench_int4, m, n, k)
        print(f"{label:35s} {r16:>12s} {r8:>18s} {ri8:>14s} {ri4:>14s}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

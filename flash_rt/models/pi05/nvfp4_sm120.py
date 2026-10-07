"""SM120 (RTX 5090) NVFP4 W4A4 GEMM primitive for the Pi0.5 RTX pipeline.

Raw-pointer, BF16 in / BF16 out, framework-neutral like ``Pi05Pipeline``
itself: the frontend quantizes each weight once (``(K, N)`` BF16 ->
NVFP4 ``(N, K)`` packed + swizzled UE4M3 block scale factors, 16 elements
per block, + one FP32 global scale) and hands the pipeline an
``Nvfp4WeightSm120`` of device pointers; the pipeline owns an
``Nvfp4ActBufferSm120`` per activation width and quantizes into it every
call, so nothing is allocated on the hot path and the GEMM is CUDA-graph
safe.

``fp4_w4a16_gemm_sm120_bf16out*`` keep a legacy "w4a16" name; both
operands are FP4. Tile variants:

- ``plain``: default ``<128,128,256>`` tile;
- ``widen``: wider tile, intended for large N;
- ``pingpong``: ``KernelTmaWarpSpecializedPingpong`` schedule.

The kernels cache per-shape arguments and workspace on first use, so
the first call at a shape must happen outside CUDA-graph capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer

Nvfp4Variant = Literal["plain", "widen", "pingpong"]

REQUIRED_KERNELS = (
    "fp4_w4a16_gemm_sm120_bf16out",
    "fp4_w4a16_gemm_sm120_bf16out_widen",
    "fp4_w4a16_gemm_sm120_bf16out_pingpong",
    "bf16_weight_to_nvfp4_swizzled",
    "quantize_bf16_to_nvfp4_swizzled",
    "nvfp4_sf_swizzled_bytes",
)


def check_nvfp4_shape(n: int, k: int) -> None:
    """Shape contract shared by the weight quantizer and the GEMM."""
    if k < 64 or k % 64 != 0 or n % 16 != 0:
        raise ValueError(f"SM120 NVFP4 requires K>=64, K%64==0, N%16==0; got N={n} K={k}")


@dataclass(frozen=True)
class Nvfp4WeightSm120:
    """Device pointers of one quantized weight: ``packed_ptr`` (N, K/2)
    uint8, ``sf_ptr`` swizzled scale factors; ``alpha`` is the FP32
    global scale the GEMM applies. The frontend keeps the memory alive."""
    packed_ptr: int
    sf_ptr: int
    alpha: float
    n: int
    k: int


@dataclass(frozen=True)
class Nvfp4ActBufferSm120:
    """Activation scratch for up to ``rows`` rows of width ``k``."""
    packed: CudaBuffer
    sf: CudaBuffer
    rows: int
    k: int


def allocate_act_buffer(fvk: ModuleType, rows: int, k: int) -> Nvfp4ActBufferSm120:
    """Allocate activation scratch for ``rows`` x ``k``."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    check_nvfp4_shape(16, k)
    packed = CudaBuffer.device_zeros(rows * k // 2, np.uint8)
    sf = CudaBuffer.device_zeros(int(fvk.nvfp4_sf_swizzled_bytes(rows, k)), np.uint8)
    return Nvfp4ActBufferSm120(packed=packed, sf=sf, rows=int(rows), k=int(k))


def quantize_act(fvk: ModuleType, x_bf16_ptr: int, act: Nvfp4ActBufferSm120,
                 m: int, stream: int) -> None:
    """Quantize ``m`` rows of a BF16 ``(m, act.k)`` activation into ``act``."""
    if not 0 < m <= act.rows:
        raise ValueError(f"m={m} outside (0, {act.rows}]")
    fvk.quantize_bf16_to_nvfp4_swizzled(
        x_bf16_ptr, act.packed.ptr.value, act.sf.ptr.value, m, act.k, stream)


def gemm_bf16out(fvk: ModuleType, variant: Nvfp4Variant, act: Nvfp4ActBufferSm120,
                 weight: Nvfp4WeightSm120, out_bf16_ptr: int, m: int, stream: int) -> None:
    """``out[m, N] (bf16) = act[m, K] @ weight[N, K]^T`` on the given tile."""
    if act.k != weight.k:
        raise ValueError(f"activation K={act.k} != weight K={weight.k}")
    if not 0 < m <= act.rows:
        raise ValueError(f"m={m} outside (0, {act.rows}]")
    if variant == "plain":
        gemm = fvk.fp4_w4a16_gemm_sm120_bf16out
    elif variant == "widen":
        gemm = fvk.fp4_w4a16_gemm_sm120_bf16out_widen
    elif variant == "pingpong":
        gemm = fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong
    else:
        raise ValueError(f"unknown NVFP4 variant {variant!r}")
    gemm(act.packed.ptr.value, weight.packed_ptr, out_bf16_ptr,
         m, weight.n, weight.k, act.sf.ptr.value, weight.sf_ptr,
         weight.alpha, stream)

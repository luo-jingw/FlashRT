"""NVFP4 numerics emulated in PyTorch, for GPUs without Blackwell FP4.

Reproduces `csrc/quantize/quantize_fp4_sfa.cu` (and its linear-layout
twin `quantize_fp4_dynamic.cu`, identical math), the quantizer every
`Nvfp4Linear` weight and activation goes through:

- blocks of 16 consecutive elements along the last (K) axis;
- `amax` = max |x| of the block, in fp32 from fp16 inputs;
- block scale = `__nv_fp8_e4m3(max(amax / 6, 1e-12))`: FP8 E4M3,
  round-to-nearest-even, saturating at 448, subnormals down to 2^-9 (a
  block whose `amax / 6` rounds to 0 dequantizes to all zeros). There is
  no per-tensor global scale; `fp4_gemm` runs with `alpha = 1`;
- element = E2M1 of `x * (1 / scale)` with the kernel's thresholds
  (`<=`: 0.25 -> 0, 0.75 -> 0.5, 1.25 -> 1, 1.75 -> 1.5, 2.5 -> 2,
  3.5 -> 3, 5.0 -> 4, else 6), so ties go to the smaller magnitude;
  a NaN (0 * inf when the scale underflowed to 0) encodes as 6;
- dequantized value = E2M1 value * scale.

Every dequantized value (<= 3 significant bits times <= 4 significant
bits, magnitude in [2^-10, 2688]) is exactly representable in fp16, so
`fake_quant_nvfp4` returns fp16 without rounding, and an fp16 GEMM with
fp32 accumulation over fake-quantized operands reproduces the NVFP4
block-scaled GEMM up to the accumulation order.

`SimNvfp4Linear` has `Nvfp4Linear`'s call interface and runs anywhere
(`precision="nvfp4_sim"`); it is an accuracy tool, not a fast path.
"""
from __future__ import annotations

import torch

from flash_rt.models.imagewam.awq import AwqScaledLinear

NVFP4_BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 `(..., K)` -> (codes uint8 `(..., K)`: bit 3 = sign, bits 0-2 =
    E2M1 index; scales float8_e4m3fn `(..., K/16)`)."""
    if x.dtype != torch.float16:
        raise ValueError(f"expected fp16, got {x.dtype}")
    k = x.shape[-1]
    if k % NVFP4_BLOCK:
        raise ValueError(f"K={k} not divisible by {NVFP4_BLOCK}")
    v = x.float().reshape(*x.shape[:-1], k // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = v.abs().amax(dim=-1, keepdim=True)
    desired = torch.clamp(amax / E2M1_MAX, min=1e-12)
    bs_q = desired.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    inv_bs = 1.0 / bs_q.float()
    q = v * inv_bs
    aq = q.abs()
    mant = torch.full_like(aq, 7, dtype=torch.uint8)
    for t in _E2M1_THRESHOLDS:
        mant -= (aq <= t).to(torch.uint8)
    sign = (q < 0).to(torch.uint8) << 3
    codes = (sign | mant).reshape(x.shape)
    return codes, bs_q.reshape(*x.shape[:-1], k // NVFP4_BLOCK)


def dequantize_nvfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of `quantize_nvfp4`, exact in fp16."""
    k = codes.shape[-1]
    # E2M1 magnitude by arithmetic, not a device lookup table: no host->device
    # copy, so this also runs inside CUDA Graph capture.
    #   index 0..3 -> 0, 0.5, 1, 1.5 = index / 2
    #   index 4..7 -> 2, 3, 4, 6     = 2^(1 + (index-4)//2) * (1 + (index-4)%2 / 2)
    m = (codes & 7).float()
    hi = torch.exp2(torch.floor((m - 4.0) * 0.5) + 1.0) * (1.0 + 0.5 * torch.remainder(m - 4.0, 2.0))
    mag = torch.where(m < 4.0, m * 0.5, hi)
    val = torch.where((codes & 8) != 0, -mag, mag)
    val = val.reshape(*codes.shape[:-1], k // NVFP4_BLOCK, NVFP4_BLOCK) * scales.float().unsqueeze(-1)
    return val.reshape(codes.shape).to(torch.float16)


def fake_quant_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """fp16 -> the fp16 values an NVFP4 GEMM actually multiplies."""
    return dequantize_nvfp4(*quantize_nvfp4(x))


def _view_fp16(ptr: int, m: int, k: int) -> torch.Tensor:
    interface = {"data": (int(ptr), False), "shape": (int(m), int(k)),
                 "typestr": "<f2", "version": 3}
    owner = type("_SimFp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device="cuda")


class SimNvfp4Linear(AwqScaledLinear):
    """`out[M,N] = fq(x)[M,K] @ fq(W)[K,N]` with `fq = fake_quant_nvfp4`
    along K for both operands (the weight as `W^T`, `(N,K)`, the layout
    `Nvfp4Linear` quantizes), fp32 accumulation (`GemmRunner.fp16_nn`),
    fp16 output. `awq_inv_s`: see `AwqScaledLinear`; the weight passed in
    must already carry the AWQ scale."""

    def __init__(self, gemm, weight_fp16_ptr: int, n: int, k: int, *,
                 awq_inv_s: torch.Tensor | None = None):
        super().__init__()
        if k % NVFP4_BLOCK:
            raise ValueError(f"NVFP4 requires K divisible by 16, got K={k}")
        self.gemm = gemm
        self.n, self.k = int(n), int(k)
        w_kn = _view_fp16(weight_fp16_ptr, self.k, self.n)
        self._w_fq = fake_quant_nvfp4(w_kn.t().contiguous()).t().contiguous()  # (K,N)
        self._awq_inv_s = awq_inv_s
        self._x_fq = None

    @property
    def awq_inv_s(self) -> torch.Tensor | None:
        return self._awq_inv_s

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        # The torch ops run on torch's current stream, which is `stream`
        # on every frontend path (eager run, warmup and capture).
        self._x_fq = fake_quant_nvfp4(_view_fp16(x_ptr, m, self.k))
        self.gemm.fp16_nn(self._x_fq.data_ptr(), self._w_fq.data_ptr(), out_ptr, m, self.n, self.k, stream)

"""PyTorch reference of the SM100 block-scaled 4-bit operand formats.

Reproduces, in plain PyTorch, what the repository's CUDA quantizers
write for the tcgen05 block-scaled GEMMs (`cutlass_fp4_gemm_variant`,
`cutlass_fp4_gemm_e0m3w`):

- Element grids, 4-bit sign-magnitude codes:
  - E2M1 (NVFP4): magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, encoded by the
    `<=` thresholds of `fp32_to_e2m1_sfa` (`csrc/quantize/quantize_fp4_sfa.cu`).
  - E0M3 (uniform INT4): magnitudes 0..7, round-half-even integer with
    clamp to 7, `-0` normalized to `+0` (`csrc/quantize/quantize_e0m3_sfa.cu`).
- One UE4M3 scale per 16 consecutive K elements, round-to-nearest-even,
  saturating at 448. Scale rules:
  - `"amax"`: `amax / qmax` (qmax 6 for E2M1, 7 for E0M3), floored at
    1e-12 before rounding. E2M1 multiplies by the fp32 reciprocal of 6
    (the kernels' `--use_fast_math` lowering); E0M3 divides exactly
    (`__fdiv_rn`).
  - `"mse"` (E2M1 only): the 9-candidate search of
    `kernel_quantize_fp4_sfa_mse`.
  Elements are multiplied by the fp32 reciprocal of the decoded scale.
- Hadamard rotation along K, block-diagonal, orthonormal Sylvester
  matrix. `fwht16_butterfly` is the in-register butterfly of
  `csrc/fused_fp4/pi05_e0m3_act.cu` (`fwht16_regs`).
- Byte layouts: two codes per byte (even K index in the low nibble), and
  the CUTLASS `Sm1xxBlockScaledConfig<16>` scale-factor layout used for
  both SFA (rows = M) and SFB (rows = N).

Measured against the CUDA kernels compiled for sm_90 from the same
sources and flags (`tools/check_blockscaled_quantizers_sm90.py`: 130M
real ImageWAM weight elements, activations with outliers, and blocks
built on the E2M1 rounding thresholds): NVFP4 amax and E0M3 (plain,
rotated, and the `e0m3_hadamard` weight preparation) match byte for
byte. The NVFP4 MSE search differs on about 2 codes per million: the
kernel accumulates its per-candidate squared error sequentially and its
PTX contracts `e * scale - v` and `err += d * d` into `fma.rn`, so
near-equal candidates can tie-break differently from `torch.sum` over
separately rounded products.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

BLOCK = 16
UE4M3_MAX = 448.0
E2M1_MAX = 6.0
E0M3_MAX = 7.0
FORMATS = ("e2m1", "e0m3")
SCALE_RULES = ("amax", "mse")

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_UPPER = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MSE_MULTS = (0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5)
_SF_ATOM_ROWS = 128
_SF_ATOM_K = 64
_SF_ATOM_BYTES = 512


@dataclass
class BlockQuantized:
    """One block-scaled operand: `codes` uint8 `[R, K]` (4-bit
    sign-magnitude codes, one per byte), `scale_bytes` uint8 `[R, K/16]`
    (UE4M3 bit patterns), `scales` float32 `[R, K/16]` (decoded scale
    values), `fmt` `"e2m1"` or `"e0m3"`."""

    codes: torch.Tensor
    scale_bytes: torch.Tensor
    scales: torch.Tensor
    fmt: str


def ue4m3_round(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest UE4M3 value (ties to even, saturating at 448,
    negative inputs clamp to 0), returned as float32."""
    return x.float().clamp(0.0, UE4M3_MAX).to(torch.float8_e4m3fn).float()


def ue4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> UE4M3 bit patterns (uint8), same rounding as `ue4m3_round`."""
    return x.float().clamp(0.0, UE4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """Scaled fp32 values -> E2M1 codes, `fp32_to_e2m1_sfa` thresholds.
    NaN and +-inf (zero scale) encode magnitude 6, as the kernel's
    comparison chain falls through to its last branch."""
    a = v.abs()
    mant = torch.full_like(v, 7, dtype=torch.uint8)
    for idx in range(len(_E2M1_UPPER) - 1, -1, -1):
        mant = torch.where(a <= _E2M1_UPPER[idx], torch.tensor(idx, dtype=torch.uint8, device=v.device), mant)
    sign = (v < 0).to(torch.uint8) << 3
    return sign | mant


def _e0m3_codes(v: torch.Tensor) -> torch.Tensor:
    """Scaled fp32 values -> E0M3 codes: round-half-even integer of
    |v|, clamp 7, sign only for a nonzero magnitude. NaN (0 * inf)
    converts to 0 and +-inf saturates to 7, as `__float2int_rn` does."""
    a = torch.nan_to_num(v.abs(), nan=0.0, posinf=float(E0M3_MAX), neginf=float(E0M3_MAX))
    mag = torch.round(a).clamp(max=E0M3_MAX).to(torch.uint8)
    sign = ((v < 0) & (mag > 0)).to(torch.uint8) << 3
    return sign | mag


def decode_codes(codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """4-bit codes -> element values (float32, before scaling)."""
    mag_idx = (codes & 7).long()
    if fmt == "e2m1":
        lut = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=codes.device)
        mag = lut[mag_idx]
    elif fmt == "e0m3":
        mag = mag_idx.float()
    else:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    return torch.where((codes & 8) != 0, -mag, mag)


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Scaled value -> E2M1 grid value (float32)."""
    return decode_codes(_e2m1_codes(v), "e2m1")


def e0m3_round(v: torch.Tensor) -> torch.Tensor:
    """Scaled value -> E0M3 grid value (float32)."""
    return decode_codes(_e0m3_codes(v), "e0m3")


def _encode(v: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "e2m1":
        return _e2m1_codes(v)
    if fmt == "e0m3":
        return _e0m3_codes(v)
    raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")


def quantize_blocks(x: torch.Tensor, fmt: str, scale_rule: str = "amax") -> BlockQuantized:
    """`x` `[R, K]` (any float dtype, K % 16 == 0) -> block-scaled codes.
    Input values are taken as float32 (the kernels read fp16 and widen)."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    if scale_rule not in SCALE_RULES:
        raise ValueError(f"scale_rule must be one of {SCALE_RULES}, got {scale_rule!r}")
    if scale_rule == "mse" and fmt != "e2m1":
        raise ValueError("scale_rule='mse' exists only for e2m1 (kernel_quantize_fp4_sfa_mse)")
    if x.dim() != 2 or x.shape[1] % BLOCK != 0:
        raise ValueError(f"x must be [R, K] with K % {BLOCK} == 0, got {tuple(x.shape)}")
    rows, k = x.shape
    xb = x.float().reshape(rows, k // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1)
    qmax = E2M1_MAX if fmt == "e2m1" else E0M3_MAX
    if scale_rule == "amax":
        scale_bytes = ue4m3_bytes(torch.clamp(_desired_scale(amax, fmt), min=1e-12))
    else:
        scale_bytes = _mse_scale_bytes(xb, amax)
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    inv = torch.reciprocal(scales).unsqueeze(-1)
    codes = _encode(xb * inv, fmt).reshape(rows, k)
    return BlockQuantized(codes=codes, scale_bytes=scale_bytes, scales=scales, fmt=fmt)


def _desired_scale(amax: torch.Tensor, fmt: str) -> torch.Tensor:
    """Unrounded block scale. The NVFP4 kernels build with
    `--use_fast_math`, which turns `amax / 6.f` into a multiplication by
    the fp32 reciprocal of 6; the E0M3 kernels use `__fdiv_rn`, an IEEE
    division (it differs exactly at the ties `amax / 7` hits often, e.g.
    amax = 1.3671875 gives exactly 25/128)."""
    if fmt == "e2m1":
        return amax * torch.tensor(1.0 / E2M1_MAX, dtype=torch.float32, device=amax.device)
    return torch.div(amax, torch.full_like(amax, E0M3_MAX))


def _mse_scale_bytes(xb: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """`kernel_quantize_fp4_sfa_mse`: 9 candidate scales around
    `amax/6`, keep the first one with the smallest squared error."""
    base = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    best_err = torch.full_like(amax, float("inf"))
    best = ue4m3_bytes(torch.full_like(amax, 1e-12))
    for mult in _MSE_MULTS:
        cand = ue4m3_bytes(torch.clamp(base * mult, min=1e-12))
        s = cand.view(torch.float8_e4m3fn).float()
        deq = e2m1_round(xb * torch.reciprocal(s).unsqueeze(-1)) * s.unsqueeze(-1)
        err = ((deq - xb) ** 2).sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best = torch.where(better, cand, best)
    return best


def dequantize_blocks(q: BlockQuantized) -> torch.Tensor:
    """Block-scaled codes -> float32 `[R, K]` values (element * scale)."""
    rows, k = q.codes.shape
    vals = decode_codes(q.codes, q.fmt).reshape(rows, k // BLOCK, BLOCK)
    return (vals * q.scales.unsqueeze(-1)).reshape(rows, k)


def fake_quantize(x: torch.Tensor, fmt: str, scale_rule: str = "amax") -> torch.Tensor:
    """Quantize then dequantize `[R, K]`; float32 result."""
    return dequantize_blocks(quantize_blocks(x, fmt, scale_rule))


def hadamard_matrix(n: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix `H_n / sqrt(n)`, `n` a power
    of two. Symmetric, so it is its own inverse."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / (n ** 0.5)).to(device=device, dtype=dtype)


def rotate_k_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    """Multiply every `block`-wide chunk of the last dim by the
    orthonormal Sylvester matrix (block-diagonal rotation along K).
    Computed in float32; returns float32."""
    k = x.shape[-1]
    if k % block != 0:
        raise ValueError(f"last dim {k} is not a multiple of block {block}")
    h = hadamard_matrix(block, device=x.device)
    xf = x.float().reshape(*x.shape[:-1], k // block, block)
    return (xf @ h).reshape(x.shape)


def fwht16_butterfly(x: torch.Tensor) -> torch.Tensor:
    """In-register 16-point fast Walsh-Hadamard transform with the final
    `* 0.25`, same butterfly order as `fwht16_regs` in
    `csrc/fused_fp4/pi05_e0m3_act.cu`. Operates on every 16-wide chunk
    of the last dim, float32."""
    k = x.shape[-1]
    if k % BLOCK != 0:
        raise ValueError(f"last dim {k} is not a multiple of {BLOCK}")
    v = x.float().reshape(-1, BLOCK).clone()
    step = 1
    while step < BLOCK:
        idx = torch.tensor([i for i in range(BLOCK) if (i & step) == 0], device=v.device)
        a = v[:, idx].clone()
        b = v[:, idx + step].clone()
        v[:, idx] = a + b
        v[:, idx + step] = a - b
        step <<= 1
    return (v * 0.25).reshape(x.shape)


def prepare_e0m3_hadamard_weight(w_nk: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Offline weight preparation of the `e0m3_hadamard` tier
    (`quant_linear.E0m3HadamardLinear`): `(N, K)` weight -> the fp16
    `(N, K)` tensor handed to `quantize_e0m3_dynamic_sfa_fp16`, and the
    GEMM `alpha`.

    Steps: 16-point butterfly per K block in fp32 (the activation kernel's
    order); multiply by `2^e`, the largest power of two that keeps the
    largest block scale `amax/7` at or below UE4M3's 448 (this moves the
    block scales out of UE4M3's subnormal range); round to fp16.
    `alpha = 2^-e` undoes the pre-scale exactly in the GEMM."""
    w_rot = fwht16_butterfly(w_nk)
    amax = float(w_rot.abs().max())
    e = math.floor(math.log2(UE4M3_MAX * E0M3_MAX / amax)) if amax > 0 else 0
    return (w_rot * (2.0 ** e)).to(torch.float16).contiguous(), float(2.0 ** -e)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """`[R, K]` 4-bit codes -> `[R, K/2]` bytes, even K index in the low
    nibble (the kernels' `lo | (hi << 4)`)."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_codes`."""
    rows, half = packed.shape
    out = torch.empty(rows, 2 * half, dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out


def sf_size_bytes(rows: int, k: int) -> int:
    """Bytes of an SFA/SFB buffer: `cosize` of
    `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SF{A,B}`, i.e. whole
    128-row x 64-K atoms of 512 bytes."""
    return -(-rows // _SF_ATOM_ROWS) * -(-k // _SF_ATOM_K) * _SF_ATOM_BYTES


def sf_offsets(rows: int, k: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Byte offset of scale `(row, kb)` (kb = K block of 16) in the
    CUTLASS scale-factor layout, int64 `[rows, K/16]`.

    Atom `((32,4),(16,4)):((16,4),(0,1))` covers 128 rows x 64 K; atoms
    are tiled K-first (`Step<_2,_1,_3>`)."""
    k_atoms = -(-k // _SF_ATOM_K)
    r = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    kb = torch.arange(k // BLOCK, device=device, dtype=torch.int64).unsqueeze(0)
    atom = (r // _SF_ATOM_ROWS) * k_atoms + (kb * BLOCK) // _SF_ATOM_K
    rin = r % _SF_ATOM_ROWS
    inner = (rin % 32) * 16 + (rin // 32) * 4 + kb % 4
    return atom * _SF_ATOM_BYTES + inner


def pack_scales(scale_bytes: torch.Tensor) -> torch.Tensor:
    """`[rows, K/16]` UE4M3 bytes -> zero-filled scale-factor buffer in
    the CUTLASS layout (padding entries stay 0, as the zero-initialized
    FlashRT buffers do)."""
    rows, kb = scale_bytes.shape
    k = kb * BLOCK
    out = torch.zeros(sf_size_bytes(rows, k), dtype=torch.uint8, device=scale_bytes.device)
    out[sf_offsets(rows, k, device=scale_bytes.device).reshape(-1)] = scale_bytes.reshape(-1)
    return out

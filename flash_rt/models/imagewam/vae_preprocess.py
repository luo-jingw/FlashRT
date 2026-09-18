"""ImageWAM VAE input preprocessing on the GPU (roadmap item 2, `plan.md`).

`VaePreprocessor` turns one or two `(H, W, 3)` uint8 camera views into
the VAE input, `(1, 3, out_h, num_views * out_w)` BF16 NCHW, normalized
to `x * 2 / 255 - 1`, with one `imagewam_vae_preprocess_bf16` launch per
view (`csrc/kernels/imagewam_vae_preprocess.cu`).

Resize modes:

- `"area"` (served default): bit-identical to
  `vae_encoder._prep_view` + `torch.cat`, i.e. torch
  `F.interpolate(mode="area")` straight to `out_hw` (no aspect-ratio
  handling), then `x * 2.0 / 255.0 - 1.0` in float32, then BF16. When
  the view is already `out_hw`, the kernel reads a 256-entry BF16
  normalization table instead.
- `"pil_bilinear"`: the resize is bit-identical to the official LIBERO
  eval's `eval_libero_single._center_crop_resize` (PIL `Image.resize(...,
  BILINEAR)` to cover `out_hw` with the aspect ratio kept, then a center
  crop); normalization is the same float32-derived table as `"area"`.
  The official eval normalizes in BF16 arithmetic instead, which differs
  from this table in 127 of 256 entries (issues.md ISSUE-030), so the
  whole preprocessing is not bit-identical to the official eval.

The normalization table is computed on the GPU with the same torch
expression `_prep_view` uses, so it equals that path bit-for-bit by
construction. PIL's resample coefficients are reproduced from
Pillow's `Resample.c` (`precompute_coeffs` + `normalize_coeffs_8bpc`,
22-bit fixed point) in double precision on the host, once per input
size, and kept on the device in a `PilResizePlan`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk

RESIZE_MODES = ("area", "pil_bilinear")

_MODE_IDENTITY = 0
_MODE_AREA = 1
_MODE_PIL_BILINEAR = 2
_PIL_PRECISION_BITS = 22
# torch computes `x / 255.0` (Python scalar divisor) as x * (1.0f / 255.0f).
_INV255_F32 = float(np.float32(1.0) / np.float32(255.0))


@dataclass(frozen=True)
class PilAxisTable:
    """PIL bilinear resample table for one axis, on the device.

    `bounds`: `(resized, 2)` int32 `{first input index, tap count}`.
    `coeffs`: `(resized, ksize)` int32 fixed-point weights.
    """
    bounds: torch.Tensor
    coeffs: torch.Tensor
    ksize: int


@dataclass(frozen=True)
class PilResizePlan:
    """PIL `_center_crop_resize` plan for one `(in_h, in_w)` input.

    An axis whose resized length equals the input length has no pass
    (`None`), matching PIL, which skips that pass.
    """
    horizontal: PilAxisTable | None
    vertical: PilAxisTable | None
    crop_left: int
    crop_top: int


def _pil_bilinear_axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Pillow `precompute_coeffs` (bilinear filter, support 1.0) followed by
    `normalize_coeffs_8bpc`, in the same double-precision order."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((out_size, 2), dtype=np.int32)
    coeffs = np.zeros((out_size, ksize), dtype=np.int32)
    ss = 1.0 / filterscale
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        weights = []
        total = 0.0
        for x in range(xmax):
            t = abs((x + xmin - center + 0.5) * ss)
            w = 1.0 - t if t < 1.0 else 0.0
            weights.append(w)
            total += w
        for x, w in enumerate(weights):
            if total != 0.0:
                w = w / total
            if w < 0:
                coeffs[xx, x] = int(-0.5 + w * (1 << _PIL_PRECISION_BITS))
            else:
                coeffs[xx, x] = int(0.5 + w * (1 << _PIL_PRECISION_BITS))
        bounds[xx, 0] = xmin
        bounds[xx, 1] = xmax
    return bounds, coeffs, ksize


class VaePreprocessor:
    """Owns the device-resident normalization table and PIL resample
    tables; launches the fused preprocessing kernel."""

    def __init__(self, *, resize: str = "area", out_hw: tuple[int, int] = (224, 224),
                 device: str = "cuda") -> None:
        if resize not in RESIZE_MODES:
            raise ValueError(f"resize={resize!r} -- must be one of {RESIZE_MODES}")
        self.resize = resize
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.device = device
        # Same expression and dtype chain as vae_encoder._prep_view.
        self.lut = (torch.arange(256, dtype=torch.float32, device=device) * 2.0 / 255.0 - 1.0).to(torch.bfloat16)
        self._pil_plans: dict[tuple[int, int], PilResizePlan] = {}

    def prepare(self, in_h: int, in_w: int) -> None:
        """Builds the device tables for an input size (PIL mode only).
        Idempotent. Call before CUDA-graph capture of `run()`."""
        if self.resize != "pil_bilinear" or (in_h, in_w) in self._pil_plans:
            return
        out_h, out_w = self.out_hw
        scale = max(out_w / in_w, out_h / in_h)
        resized_w, resized_h = round(in_w * scale), round(in_h * scale)
        crop_left = max((resized_w - out_w) // 2, 0)
        crop_top = max((resized_h - out_h) // 2, 0)
        if crop_left + out_w > resized_w or crop_top + out_h > resized_h:
            raise ValueError(f"center crop {self.out_hw} does not fit resized ({resized_h},{resized_w})")

        def axis(in_size: int, resized: int) -> PilAxisTable | None:
            if resized == in_size:
                return None
            b, c, k = _pil_bilinear_axis(in_size, resized)
            return PilAxisTable(bounds=torch.from_numpy(b).to(self.device),
                                coeffs=torch.from_numpy(c).to(self.device), ksize=k)

        self._pil_plans[(in_h, in_w)] = PilResizePlan(
            horizontal=axis(in_w, resized_w), vertical=axis(in_h, resized_h),
            crop_left=crop_left, crop_top=crop_top)

    def run(self, views: Sequence[torch.Tensor], out: torch.Tensor, stream: int) -> None:
        """`views`: `(H, W, 3)` uint8 CUDA contiguous tensors. `out`:
        `(1, 3, out_h, len(views) * out_w)` BF16 CUDA contiguous."""
        out_h, out_w = self.out_hw
        total_w = len(views) * out_w
        if out.dtype != torch.bfloat16 or not out.is_contiguous() or tuple(out.shape) != (1, 3, out_h, total_w):
            raise ValueError(f"out must be contiguous BF16 (1,3,{out_h},{total_w}), got "
                             f"{tuple(out.shape)} {out.dtype}")
        for i, view in enumerate(views):
            if view.dtype != torch.uint8 or view.ndim != 3 or view.shape[-1] != 3 or not view.is_cuda \
                    or not view.is_contiguous():
                raise ValueError(f"view {i} must be a contiguous (H,W,3) uint8 CUDA tensor, got "
                                 f"shape={tuple(view.shape)} dtype={view.dtype} device={view.device}")
            in_h, in_w = int(view.shape[0]), int(view.shape[1])
            h_b = h_c = v_b = v_c = 0
            h_k = v_k = crop_left = crop_top = 0
            if (in_h, in_w) == self.out_hw:
                mode = _MODE_IDENTITY
            elif self.resize == "area":
                mode = _MODE_AREA
            else:
                mode = _MODE_PIL_BILINEAR
                self.prepare(in_h, in_w)
                plan = self._pil_plans[(in_h, in_w)]
                if plan.horizontal is not None:
                    h_b, h_c, h_k = (plan.horizontal.bounds.data_ptr(), plan.horizontal.coeffs.data_ptr(),
                                     plan.horizontal.ksize)
                if plan.vertical is not None:
                    v_b, v_c, v_k = (plan.vertical.bounds.data_ptr(), plan.vertical.coeffs.data_ptr(),
                                     plan.vertical.ksize)
                crop_left, crop_top = plan.crop_left, plan.crop_top
            rc = fvk.imagewam_vae_preprocess_bf16(
                view.data_ptr(), self.lut.data_ptr(), out.data_ptr(),
                in_h, in_w, out_h, out_w, total_w, i * out_w, mode,
                h_b, h_c, h_k, crop_left, v_b, v_c, v_k, crop_top,
                _INV255_F32, stream)
            if rc != 0:
                raise RuntimeError(f"imagewam_vae_preprocess_bf16 failed rc={rc} (view {i}, {in_h}x{in_w})")

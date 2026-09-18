"""Per-GEMM-input activation statistics from a real ImageWAM forward pass.

`pipeline_thor.py` dispatches every weight-projection GEMM through
`weights[key](x_ptr, out_ptr, m, stream)`. `ActivationRecorder.wrap()`
returns a copy of a weight dict in which every such callable whose
input is fp16 is replaced by a `_RecordingLinear` that reads the
`(m, K)` fp16 input, updates this sample's statistics for that site, and
then calls the original. Raw-pointer entries (norm scales, biases) and
`Bf16OutLinear` (`txt_in`/`img_in`, BF16 input, never quantized) are
passed through unchanged.

Per call, per site (on the GPU): absmax, the 99 / 99.9 / 99.99th
percentiles of |x|, and per-input-channel absmax (length K, the AWQ
statistic). Within one sample the site's calls are reduced by max:
ActionDiT sites run once per denoise step, and the house rule
(`docs/calibration.md` §4.2) calibrates one scale across all steps.
Across samples, `calibration_file.build_calibration` applies the house
percentile reducer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flash_rt.models.imagewam.quant_linear import Bf16OutLinear, CutlassFp16SwiGluMlp, Nvfp4SwiGluMlp

ABS_PERCENTILES = (99.0, 99.9, 99.99)
_QUANTILE_MAX_ELEMENTS = 1 << 24  # torch.quantile's input size limit


def site_name(key: tuple) -> str:
    """Weight-dict key -> the calibration file's site name:
    `("backbone", "single", 3, "linear1.weight")` ->
    `"backbone.single.3.linear1.weight"`."""
    return ".".join(str(part) for part in key)


@dataclass
class SiteStats:
    """One site's statistics for one sample (max over its calls)."""
    absmax: float
    abs_percentiles: np.ndarray   # (len(ABS_PERCENTILES),) float32
    channel_amax: np.ndarray      # (K,) float32
    calls: int
    rows: int                     # m of the site's GEMM


@dataclass
class SampleStats:
    sites: dict[str, SiteStats]


def _view_fp16(ptr: int, m: int, k: int) -> torch.Tensor:
    interface = {"data": (int(ptr), False), "shape": (int(m), int(k)),
                 "typestr": "<f2", "version": 3}
    owner = type("_RecorderFp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device="cuda")


class _SiteAccumulator:
    def __init__(self, k: int):
        self.k = k
        self.absmax = torch.zeros((), dtype=torch.float32, device="cuda")
        self.pct = torch.zeros(len(ABS_PERCENTILES), dtype=torch.float32, device="cuda")
        self.channel = torch.zeros(k, dtype=torch.float32, device="cuda")
        self.calls = 0
        self.rows = 0

    def observe(self, x: torch.Tensor) -> None:
        ax = x.float().abs()
        ch = ax.amax(dim=0)
        torch.maximum(self.channel, ch, out=self.channel)
        torch.maximum(self.absmax, ch.max(), out=self.absmax)
        flat = ax.flatten()
        if flat.numel() > _QUANTILE_MAX_ELEMENTS:
            raise ValueError(f"site input has {flat.numel()} elements, above torch.quantile's limit")
        q = torch.tensor([p / 100.0 for p in ABS_PERCENTILES], dtype=torch.float32, device="cuda")
        torch.maximum(self.pct, torch.quantile(flat, q), out=self.pct)
        self.calls += 1
        self.rows = x.shape[0]

    def finish(self) -> SiteStats:
        return SiteStats(absmax=float(self.absmax.item()),
                         abs_percentiles=self.pct.cpu().numpy().astype(np.float32),
                         channel_amax=self.channel.cpu().numpy().astype(np.float32),
                         calls=self.calls, rows=self.rows)


class _RecordingLinear:
    """Same call interface as every `quant_linear.py` class; records the
    input, then runs the wrapped op unchanged."""

    def __init__(self, inner, name: str, recorder: "ActivationRecorder"):
        self.inner = inner
        self.name = name
        self.k = int(inner.k)
        self._recorder = recorder

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self._recorder._observe(self.name, _view_fp16(x_ptr, m, self.k))
        self.inner(x_ptr, out_ptr, m, stream)


class ActivationRecorder:
    """Records per-site statistics for one sample at a time:
    `begin_sample()` -> run the pipeline with `wrap(weights)` ->
    `end_sample()`."""

    def __init__(self):
        self._acc: dict[str, _SiteAccumulator] | None = None

    def wrap(self, weights: dict) -> dict:
        wrapped = {}
        for key, value in weights.items():
            if isinstance(value, (CutlassFp16SwiGluMlp, Nvfp4SwiGluMlp)):
                # pipeline_thor._mlp_gate_up dispatches these by class;
                # a wrapper would route them down the merged-buffer path.
                raise ValueError(f"{site_name(key)}: record a precision='fp16' frontend "
                                 f"(fused SwiGLU classes cannot be wrapped)")
            if isinstance(value, int) or isinstance(value, Bf16OutLinear):
                wrapped[key] = value
            else:
                wrapped[key] = _RecordingLinear(value, site_name(key), self)
        return wrapped

    def begin_sample(self) -> None:
        self._acc = {}

    def _observe(self, name: str, x: torch.Tensor) -> None:
        if self._acc is None:
            raise RuntimeError("ActivationRecorder: call begin_sample() before running the pipeline")
        acc = self._acc.get(name)
        if acc is None:
            acc = self._acc[name] = _SiteAccumulator(x.shape[1])
        acc.observe(x)

    def end_sample(self) -> SampleStats:
        if self._acc is None:
            raise RuntimeError("ActivationRecorder: end_sample() without begin_sample()")
        torch.cuda.synchronize()
        stats = SampleStats(sites={n: a.finish() for n, a in self._acc.items()})
        self._acc = None
        return stats

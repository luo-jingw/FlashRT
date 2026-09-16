"""Real `dataset_stats.json` min/max normalization -- closed-loop
real-robot conditioning (proprio in, action out), found 2026-09-15
while scoping real closed-loop testing (opportunities.md).

Ported EXACTLY from the real training-side normalizer
(`imagewam/datasets/lerobot/utils/normalizer.py`'s own
`SingleFieldLinearNormalizer`, mode="min/max" -- confirmed as the mode
THIS release actually uses via its own `config.yaml`:
`use_stepwise_action_norm: false`, `norm_default_mode: min/max`,
`norm_exception_mode: null` -- so BOTH `state` (proprio) and `action`
use `global_min`/`global_max` (never `stepwise_*`), never `q01/q99` or
`z-score`, for this specific checkpoint). A DIFFERENT release could use
a different mode -- this module hardcodes min/max because that's what
was actually confirmed for the real LIBERO release this project
targets, not because it's the only mode that exists; check a new
release's own `config.yaml` before reusing this against it unchanged.

`dataset_stats.json` sits alongside `model.pt` in the release
directory (`state`/`action`, each `{"default": {...various stat
arrays...}}`) -- `state["default"]` is `proprio_dim`-wide (8 for this
release), `action["default"]` is `action_dim`-wide (7).
"""
from __future__ import annotations

import json

import torch

DEV = "cuda"
F32 = torch.float32

_STD_REG = 1e-8  # unused for min/max mode, kept for parity with the real normalizer's own constant
_RANGE_TOL = 1e-4
_OUTPUT_MIN, _OUTPUT_MAX = -1.0, 1.0
_CLAMP = 5.0


def load_dataset_stats(dataset_stats_path: str) -> dict:
    with open(dataset_stats_path) as f:
        return json.load(f)


def _min_max_scale_offset(global_min: torch.Tensor, global_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real `SingleFieldLinearNormalizer.__init__`'s own min/max branch,
    verbatim (including the `range_tol`/`ignore_dim` degenerate-range
    handling for a dim whose real training data never varied)."""
    input_range = global_max - global_min
    ignore_dim = input_range < _RANGE_TOL
    input_range = torch.where(ignore_dim, torch.full_like(input_range, _OUTPUT_MAX - _OUTPUT_MIN), input_range)
    scale = (_OUTPUT_MAX - _OUTPUT_MIN) / input_range
    offset = _OUTPUT_MIN - scale * global_min
    offset = torch.where(ignore_dim, (_OUTPUT_MAX + _OUTPUT_MIN) / 2 - global_min, offset)
    return scale, offset


class MinMaxNormalizer:
    """One field (`state` or `action`)'s real min/max scale/offset,
    resident on `device` -- `forward()` normalizes a real-unit tensor
    into the model's own [-1,1] training space (used for `proprio`
    INPUT); `backward()` denormalizes the model's own output back into
    real units (used for `action` OUTPUT). Matches
    `SingleFieldLinearNormalizer.forward`/`.backward` exactly, including
    the real `[-5,5]` forward clamp.
    """

    def __init__(self, global_min: list[float], global_max: list[float], *, device: str = DEV):
        gmin = torch.tensor(global_min, dtype=F32, device=device)
        gmax = torch.tensor(global_max, dtype=F32, device=device)
        self.scale, self.offset = _min_max_scale_offset(gmin, gmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=F32, device=self.scale.device) * self.scale + self.offset
        return torch.clamp(x, -_CLAMP, _CLAMP)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(dtype=F32, device=self.offset.device) - self.offset) / self.scale


def load_real_normalizers(dataset_stats_path: str, *, device: str = DEV) -> tuple[MinMaxNormalizer, MinMaxNormalizer]:
    """`(state_normalizer, action_normalizer)` -- `state` normalizes
    real proprio INTO the model's space (`.forward()`); `action`
    denormalizes the model's flow-matching output back OUT to real
    units (`.backward()`). Both keyed `"default"` in the real
    `dataset_stats.json` (this release's only embodiment)."""
    stats = load_dataset_stats(dataset_stats_path)
    state = stats["state"]["default"]
    action = stats["action"]["default"]
    state_norm = MinMaxNormalizer(state["global_min"], state["global_max"], device=device)
    action_norm = MinMaxNormalizer(action["global_min"], action["global_max"], device=device)
    return state_norm, action_norm

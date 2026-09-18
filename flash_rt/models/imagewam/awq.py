"""AWQ per-input-channel scaling for ImageWAM's NVFP4 GEMMs.

Math (the house convention, Pi0.5's `_awq_scale_weight` in
`flash_rt/frontends/torch/pi05_thor_fp4.py`):

    a[k]  = per-input-channel |x| max over the calibration samples
            (`calibration_file.SiteCalibration.channel_amax`)
    s[k]  = clamp((a[k] / mean(a)) ** alpha, 0.25, 4.0)
    W'    = W * s   (row k of the `(K, N)` weight times s[k]; W' is what
                     gets quantized to NVFP4)
    x'    = x / s   (must be what reaches the GEMM input)
    x' @ W' = x @ W in exact arithmetic.

Channels with large activations get a larger weight and a smaller
activation, so they stop dominating their 16-element activation block
and their weights are quantized relative to a larger block maximum.

Where `x / s` comes from, with no extra kernel (each fold is an exact
algebraic identity; only the fp16 storage of the folded constants
rounds, measured in `tests/test_imagewam_awq.py`):

A. AdaLN-fed GEMMs, `x = LN(h) * (1 + scale) + shift` (the
   `ada_layer_norm_*` kernels read fp16 `scale`/`shift` vectors):
       x / s = LN(h) * (1 + scale') + shift',
       1 + scale' = (1 + scale) / s,   shift' = shift / s.
   Sites: backbone double `{txt,img}_qkv` (modulation 1) and
   `{txt,img}_mlp0` (modulation 2); backbone single `linear1`; ActionDiT
   double `qkv`, `mlp0`; ActionDiT single `linear1`. All layers of a
   stream share one modulation, so each AWQ layer gets its own folded
   pair, computed once per (layer, modulation) and cached
   (`AwqScaledLinear.folded_modulation`): graph warmup fills the cache
   before capture, so replay only reads it.
B. Down projections, `x = silu(g) * u` with `u` a column block of the
   preceding merged GEMM (`mlp0 = [gate | up]`,
   `linear1 = [q | k | v | gate | up]`):
       x / s = silu(g) * (u / s),
   so the preceding weight's up columns are multiplied by `1/s` before
   that weight is quantized (Pi0.5 folds its down-projection `inv_s`
   into the up rows the same way). Sites: backbone double
   `{txt,img}_mlp2`, single `mlp_down`; ActionDiT double `mlp2`, single
   `mlp_down`.

No exact fold point: `proj` / `attn_out_proj` (input = attention output;
V is shared by both backbone streams and by the ActionDiT's joint
attention, so scaling V columns for one consumer changes the others).
These stay unscaled.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

AWQ_S_MIN = 0.25
AWQ_S_MAX = 4.0
AWQ_SCOPES = ("adaln", "adaln+down")

# Fold A: slot names whose GEMM input is an AdaLN-modulated activation.
_ADALN_SLOTS = {
    ("backbone", "double"): ("txt_qkv.weight", "img_qkv.weight", "txt_mlp0.weight", "img_mlp0.weight"),
    ("backbone", "single"): ("linear1.weight",),
    ("action_dit", "double"): ("qkv.weight", "mlp0.weight"),
    ("action_dit", "single"): ("linear1.weight",),
}
# Fold B: down-projection slot -> preceding merged gate/up slot.
_DOWN_SLOTS = {
    ("backbone", "double"): {"txt_mlp2.weight": "txt_mlp0.weight", "img_mlp2.weight": "img_mlp0.weight"},
    ("backbone", "single"): {"mlp_down.weight": "linear1.weight"},
    ("action_dit", "double"): {"mlp2.weight": "mlp0.weight"},
    ("action_dit", "single"): {"mlp_down.weight": "linear1.weight"},
}


def awq_scale(channel_amax: torch.Tensor, alpha: float) -> torch.Tensor:
    """`s = clamp((a / mean(a)) ** alpha, 0.25, 4)`, fp32, same device."""
    a = channel_amax.float().clamp(min=1e-6)
    return (a / a.mean()).pow(alpha).clamp(min=AWQ_S_MIN, max=AWQ_S_MAX)


def fold_inv_scale_into_modulation(shift: torch.Tensor, scale: torch.Tensor,
                                   inv_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold A. `shift`/`scale`: the AdaLN modulation as produced by
    `adaln.modulation` (fp32, any shape ending in `dim`); `inv_s` = 1/s,
    fp32 `(dim,)`. Returns contiguous fp16 `(dim,)` `(shift', scale')`
    with `1 + scale' = (1 + scale) * inv_s`, `shift' = shift * inv_s`,
    computed in fp32 and rounded once to fp16."""
    sc = scale.reshape(-1).float()
    sh = shift.reshape(-1).float()
    scale_f = (1.0 + sc) * inv_s - 1.0
    shift_f = sh * inv_s
    return shift_f.to(torch.float16).contiguous(), scale_f.to(torch.float16).contiguous()


class AwqScaledLinear(ABC):
    """A GEMM whose weight carries an AWQ input scale `s` (fold A): the
    caller must feed it `x / s`. `pipeline_thor.py` folds `awq_inv_s`
    into the AdaLN modulation that produces this GEMM's input.
    `awq_inv_s is None` means no fold A for this weight (fold B needs
    no caller action). Subclasses call `super().__init__()`."""

    def __init__(self) -> None:
        self._awq_fold_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    @abstractmethod
    def awq_inv_s(self) -> torch.Tensor | None:
        """fp32 `(K,)` = 1/s on the GPU, or None."""

    def folded_modulation(self, shift: torch.Tensor,
                          scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold A for one modulation pair, computed on first use and then
        returned from a cache keyed by the pair's storage (the frontend
        owns the modulation tensors for its lifetime)."""
        inv_s = self.awq_inv_s
        if inv_s is None:
            raise RuntimeError("folded_modulation() on a weight without an AWQ input scale")
        key = (shift.data_ptr(), scale.data_ptr())
        hit = self._awq_fold_cache.get(key)
        if hit is None:
            hit = self._awq_fold_cache[key] = fold_inv_scale_into_modulation(shift, scale, inv_s)
        return hit


@dataclass
class AwqWeightPlan:
    """How one `(K, N)` weight is transformed. `input_scale`: `s`, rows
    multiplied by it (fold A or the down projection's own scale).
    `up_inv_scale`/`up_offset`: 1/s of the FOLLOWING down projection,
    multiplied into columns `[up_offset, up_offset + len)` (fold B).
    `fold_input`: whether the caller must fold `1/input_scale` into the
    AdaLN modulation (fold A); false for down projections, whose input
    is already divided through fold B."""
    input_scale: torch.Tensor | None
    fold_input: bool
    up_inv_scale: torch.Tensor | None
    up_offset: int


def plan_awq(keys: list[tuple], channel_amax: dict[tuple, torch.Tensor], dims: dict, *,
             alpha: float, scope: str) -> dict[tuple, AwqWeightPlan]:
    """AWQ plan for every weight key that gets one. `channel_amax[key]` is
    the calibration statistic of that key's GEMM input. Needs the merged
    single-stream `linear1` layout (`dims["merge_qkv_mlp"]`)."""
    if scope not in AWQ_SCOPES:
        raise ValueError(f"scope={scope!r} -- must be one of {AWQ_SCOPES}")
    if not dims.get("merge_qkv_mlp"):
        raise ValueError("AWQ fold B targets the merged single-stream linear1; merge_qkv_mlp must be set")
    up_offset = {
        ("backbone", "double"): dims["mlp_hidden"],
        ("backbone", "single"): 3 * dims["hidden"] + dims["mlp_hidden"],
        ("action_dit", "double"): dims["action_mlp_hidden"],
        ("action_dit", "single"): 3 * dims["action_attn_width"] + dims["action_mlp_hidden"],
    }
    keyset = set(keys)
    plans: dict[tuple, AwqWeightPlan] = {}
    for key in keys:
        model, stream, layer, slot = key
        if slot in _ADALN_SLOTS.get((model, stream), ()):
            plans[key] = AwqWeightPlan(input_scale=awq_scale(channel_amax[key], alpha), fold_input=True,
                                       up_inv_scale=None, up_offset=0)
    if scope == "adaln+down":
        for key in keys:
            model, stream, layer, slot = key
            prev_slot = _DOWN_SLOTS.get((model, stream), {}).get(slot)
            if prev_slot is None:
                continue
            prev = (model, stream, layer, prev_slot)
            if prev not in keyset:
                raise ValueError(f"{key}: preceding gate/up weight {prev} missing")
            s_dn = awq_scale(channel_amax[key], alpha)
            plans[key] = AwqWeightPlan(input_scale=s_dn, fold_input=False, up_inv_scale=None, up_offset=0)
            p = plans.get(prev) or AwqWeightPlan(input_scale=None, fold_input=False,
                                                 up_inv_scale=None, up_offset=0)
            p.up_inv_scale = 1.0 / s_dn
            p.up_offset = up_offset[(model, stream)]
            plans[prev] = p
    return plans


def apply_awq_plan(w_kn: torch.Tensor, plan: AwqWeightPlan) -> torch.Tensor:
    """New fp16 `(K, N)` weight: rows times `input_scale`, then the up
    columns times `up_inv_scale`, in fp32, rounded once to fp16."""
    w = w_kn.float()
    if plan.input_scale is not None:
        w = w * plan.input_scale.to(w.device).unsqueeze(1)
    if plan.up_inv_scale is not None:
        n_up = plan.up_inv_scale.numel()
        w[:, plan.up_offset:plan.up_offset + n_up] *= plan.up_inv_scale.to(w.device).unsqueeze(0)
    return w.to(torch.float16).contiguous()

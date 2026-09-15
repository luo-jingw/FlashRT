"""ImageWAM/FLUX.2 real AdaLN modulation (time-embedding-driven shift/
scale/gate), found missing alongside RoPE/QK-Norm/the real mask while
starting OPT-002's real-checkpoint work.

Real math, ported from `black-forest-labs/flux2`'s `src/flux2/model.py`
at the pinned commit `50fe5162777813d869182b139e83b10743caef15` (fetched
and read directly): `timestep_embedding` (sinusoidal), `MLPEmbedder`
(`time_in`: Linear->SiLU->Linear, no bias), `Modulation` (`silu(vec)`
projected then chunked into shift/scale/gate tuples).

**Deliberately plain PyTorch, not new FlashRT kernels**, for the
embedding/modulation computation itself: `vec` and the resulting
shift/scale/gate values are computed from a SINGLE per-batch vector
(`timestep`), not per-token -- for this project's real serving shape
(one observation processed at a time, B=1), this is a handful of KB of
GEMM work, negligible next to the backbone's own per-token GEMMs.
Forcing it through custom kernels would add real risk (new kernel math
to verify) for no measurable speed benefit. The one piece that DOES
operate on the full (S, D) hidden state -- the real LayerNorm
(`elementwise_affine=False`) each modulation step normalizes before
applying shift/scale -- uses the EXISTING, already-verified
`layer_norm_no_affine_fp16` kernel (confirmed to compute exactly
`(x-mean)*rsqrt(var+eps)` by reading `csrc/kernels/norm.cu` directly,
matching PyTorch's own `nn.LayerNorm(elementwise_affine=False)`).

Real per-block usage (not yet wired here, tracked as still-open in
opportunities.md): each DoubleStreamBlock/SingleStreamBlock consumes
its OWN Modulation instance's shift/scale/gate output; this module
gives the primitives (`timestep_embedding_real`, `mlp_embedder`,
`modulation`, `apply_modulation`, `apply_gated_residual`), not a full
block forward.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk


def timestep_embedding_real(t: torch.Tensor, dim: int = 256,
                             max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Port of the real `timestep_embedding`. t: (B,) -> (B, dim)."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t.dtype if torch.is_floating_point(t) else torch.float32)


def mlp_embedder(x: torch.Tensor, in_weight: torch.Tensor, out_weight: torch.Tensor) -> torch.Tensor:
    """Port of the real `MLPEmbedder.forward` (`disable_bias=True`, so no
    bias anywhere -- matches `time_in`'s real construction in Flux2.__init__).
    x: (B, in_dim). in_weight: (hidden_dim, in_dim). out_weight: (hidden_dim, hidden_dim)."""
    h = F.linear(x, in_weight)
    h = F.silu(h)
    return F.linear(h, out_weight)


def modulation(vec: torch.Tensor, lin_weight: torch.Tensor, double: bool):
    """Port of the real `Modulation.forward` (`disable_bias=True` for
    every real call site in Flux2.__init__ -- `double_stream_modulation_img/txt`,
    `single_stream_modulation`). vec: (B, dim). lin_weight: (multiplier*dim, dim).
    Returns (shift1,scale1,gate1) or ((shift1,scale1,gate1),(shift2,scale2,gate2))."""
    multiplier = 6 if double else 3
    out = F.linear(F.silu(vec), lin_weight)
    if out.ndim == 2:
        out = out[:, None, :]  # (B, 1, multiplier*dim) -- broadcasts over seq
    chunks = out.chunk(multiplier, dim=-1)
    return (chunks[:3], chunks[3:]) if double else (chunks[:3], None)


def head_modulation(vec: torch.Tensor, lin_weight: torch.Tensor):
    """Port of the real `Flux2ActionHead.adaLN_modulation` (OPT-001,
    `imagewam/models/backbones/action_dit_flux2.py`, read directly):
    `nn.Sequential(SiLU, Linear(hidden_dim, 2*hidden_dim, bias=False))`
    -- shift/scale ONLY, no gate, since this is a FINAL output layer
    (`Flux2ActionHead.forward`: `linear((1+scale)*norm_final(x)+shift)`),
    not a residual block -- there is nothing to gate. Same shape as
    `modulation()` above with `multiplier=2`, which that function
    doesn't support (only 3 or 6), hence this separate small helper
    rather than a `modulation()` parameter.
    vec: (B, dim). lin_weight: (2*dim, dim). Returns (shift, scale)."""
    out = F.linear(F.silu(vec), lin_weight)
    if out.ndim == 2:
        out = out[:, None, :]
    shift, scale = out.chunk(2, dim=-1)
    return shift, scale


def layer_norm_no_affine_fp16(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Real `elementwise_affine=False` LayerNorm via the existing,
    already-verified `layer_norm_no_affine_fp16` FlashRT kernel.
    x: (seq, dim) fp16, contiguous."""
    seq, dim = x.shape
    out = torch.empty_like(x)
    fvk.layer_norm_no_affine_fp16(x.data_ptr(), out.data_ptr(), seq, dim, eps, 0)
    return out


def apply_modulation(x_normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Real `(1 + scale) * norm(x) + shift`, broadcasting shift/scale
    ((1,1,dim) or (dim,)) over the sequence dimension."""
    return (1 + scale) * x_normed + shift


def apply_gated_residual(residual: torch.Tensor, sublayer_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Real `residual + gate * sublayer_out`, broadcasting gate over the
    sequence dimension."""
    return residual + gate * sublayer_out

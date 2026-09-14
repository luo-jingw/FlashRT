"""ImageWAM/FLUX.2 real RoPE (rotary position embedding) precompute.

**Real math, ported from the actual upstream source** --
`black-forest-labs/flux2` at the exact commit ImageWAM's own
`docs/dependencies.md` pins (`50fe5162777813d869182b139e83b10743caef15`),
fetched and read directly (`src/flux2/model.py`'s `rope()`, `EmbedND`,
`apply_rope`, `Flux2Params`/`Klein4BParams`), not guessed or assumed by
analogy. This confirmed a real gap: every attention kernel in this
project before this file applies NO rotary position embedding at all --
a correctness gap found while starting OPT-002's real-checkpoint
accuracy-validation work, tracked in opportunities.md.

FLUX.2 uses a 4-axis multi-dimensional RoPE (`axes_dim=[32,32,32,32]`
summing to `head_dim=128`), NOT the single-axis 1D RoPE this project's
existing `rope_apply` (csrc/kernels/rope.cu, used by other models) was
built for -- position is encoded per-axis: axis 0 = a constant "time"
value, axis 1 = image row index, axis 2 = image column index, axis 3 =
a plain running index (used by text tokens only; images leave it 0,
text leaves axes 0-2 at 0). Each axis independently rotates its own
32-element chunk of `head_dim` using its own axis's position value --
confirmed by reading `EmbedND.forward`'s `torch.cat(..., dim=-3)` over
per-axis `rope()` outputs, each computed with that axis's own
`axes_dim[i]=32` (not the full 128) as its frequency-scale denominator.

This module computes the FINAL interleaved (seq, head_dim) cos/sin
table this project's own `rope_apply_fp16_perhead` kernel
(csrc/kernels/rope.cu) expects -- table[pos, 2*d] = cos, [pos, 2*d+1] =
sin for pair d -- entirely in plain PyTorch, once per (image
resolution, text length) combination, NOT per denoise step (RoPE
tables depend only on sequence position/geometry, identical across
every one of a denoise loop's steps for a fixed observation). Verified
against the REAL upstream `rope()`/`apply_rope`/`EmbedND` functions
directly in `tests/test_imagewam_rope_kernel.py` (not just this
module's own re-derivation of the formula).
"""
from __future__ import annotations

import torch

# Klein4BParams, the real ImageWAM FLUX.2-4B variant this project targets.
FLUX2_AXES_DIM = (32, 32, 32, 32)
FLUX2_ROPE_THETA = 2000


def _rope_table(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Port of the real `rope()` from flux2/model.py. pos: (seq,).
    Returns (seq, dim/2, 2) -- [..., 0]=cos, [..., 1]=sin per pair,
    already reduced from the real code's (seq, dim/2, 2, 2) rotation-
    matrix form since only the first row (cos, -sin) and the (sin, cos)
    combination this project's kernel needs are used; see
    `_flatten_interleaved` below for how these become the kernel's
    expected interleaved layout."""
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("n,d->nd", pos.float(), omega)  # (seq, dim/2)
    return torch.stack([torch.cos(out), torch.sin(out)], dim=-1)  # (seq, dim/2, 2)


def _flatten_interleaved(cos_sin: torch.Tensor) -> torch.Tensor:
    """(seq, dim/2, 2) -> (seq, dim) interleaved [cos0,sin0,cos1,sin1,...],
    the exact layout `rope_apply_fp16_perhead` reads."""
    seq, half_dim, _ = cos_sin.shape
    return cos_sin.reshape(seq, half_dim * 2)


def build_txt_ids(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Real `build_txt_ids` convention: axis 3 = running index, axes 0-2 = 0."""
    ids = torch.zeros(seq_len, 4, dtype=torch.float32, device=device)
    ids[:, 3] = torch.arange(seq_len, dtype=torch.float32, device=device)
    return ids


def build_img_ids(token_height: int, token_width: int, time_value: float,
                   device: str = "cuda") -> torch.Tensor:
    """Real `build_img_ids` convention: axis 0 = constant time_value,
    axis 1 = row index, axis 2 = column index, axis 3 = 0."""
    ids = torch.zeros(token_height, token_width, 4, dtype=torch.float32, device=device)
    ids[..., 0] = float(time_value)
    ids[..., 1] = torch.arange(token_height, dtype=torch.float32, device=device)[:, None]
    ids[..., 2] = torch.arange(token_width, dtype=torch.float32, device=device)[None, :]
    return ids.reshape(token_height * token_width, 4)


def build_action_ids(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Real `ActionDiTFlux2.build_action_ids` convention (ImageWAM's own
    source, `action_dit_flux2.py`, not `flux2/model.py`): axis 0 = a
    constant 2.0 (a "type marker" distinguishing action tokens from text
    (axis 3) or image (axes 0-2, time_value 10.0 for ref) in the shared
    RoPE space), axis 1 = running index, axes 2-3 = 0."""
    ids = torch.zeros(seq_len, 4, dtype=torch.float32, device=device)
    ids[:, 0] = 2.0
    ids[:, 1] = torch.arange(seq_len, dtype=torch.float32, device=device)
    return ids


def embed_nd_interleaved(ids: torch.Tensor, axes_dim=FLUX2_AXES_DIM,
                          theta: int = FLUX2_ROPE_THETA) -> torch.Tensor:
    """Real `EmbedND.forward` port: ids (seq, len(axes_dim)) -> (seq, head_dim)
    interleaved cos/sin table (head_dim = sum(axes_dim)), ready for
    `rope_apply_fp16_perhead`. Each axis rotates its own contiguous
    chunk of head_dim (axis i covers elements
    [2*sum(axes_dim[:i]), 2*sum(axes_dim[:i+1]))), using ITS OWN
    axes_dim[i] as the frequency-scale denominator -- confirmed by
    reading the real `rope(ids[..., i], self.axes_dim[i], self.theta)`
    call directly, not assumed."""
    per_axis = [_flatten_interleaved(_rope_table(ids[:, i], axes_dim[i], theta))
                for i in range(len(axes_dim))]
    return torch.cat(per_axis, dim=-1).contiguous()  # (seq, sum(axes_dim))


def build_backbone_rope_table(x0: int, ref_h: int, ref_w: int, *,
                               ref_time_value: float = 10.0,
                               axes_dim=FLUX2_AXES_DIM,
                               theta: int = FLUX2_ROPE_THETA,
                               device: str = "cuda") -> torch.Tensor:
    """Combined [txt (x0 tokens) | ref image (ref_h*ref_w tokens)] RoPE
    table for ImageWAM's real `infer_action_flux2` inference path
    (target image length is always 0 there -- see opportunities.md's
    real-mask correction). `ref_time_value=10.0` matches the real
    constant `_encode_flux2_image_tokens(input_image, time_value=10.0)`
    uses for the action-inference reference image in `imagewam.py`.

    Returns (x0 + ref_h*ref_w, head_dim) interleaved cos/sin, fp16,
    ready for `rope_apply_fp16_perhead` on both Q and K.
    """
    txt_ids = build_txt_ids(x0, device=device)
    img_ids = build_img_ids(ref_h, ref_w, ref_time_value, device=device)
    ids = torch.cat([txt_ids, img_ids], dim=0)
    table = embed_nd_interleaved(ids, axes_dim=axes_dim, theta=theta)
    return table.to(torch.float16).contiguous()


def build_action_rope_table(action_len: int, *,
                             axes_dim=FLUX2_AXES_DIM,
                             theta: int = FLUX2_ROPE_THETA,
                             device: str = "cuda") -> torch.Tensor:
    """RoPE table for ActionDiT's own fresh Q/K (real
    `action_pe = video_expert.transformer.pe_embedder(action_ids)` in
    `mot.py`'s `forward_flux2_action_with_video_cache`, confirmed by
    reading that function directly -- action uses the SAME `pe_embedder`
    config as the backbone, just its own `build_action_ids` position
    convention). Returns (action_len, head_dim) interleaved cos/sin,
    fp16, ready for `rope_apply_fp16_perhead` on action's own Q/K only
    (the cached backbone K was already rotated during backbone prefill
    with `build_backbone_rope_table` and never needs re-rotating)."""
    ids = build_action_ids(action_len, device=device)
    table = embed_nd_interleaved(ids, axes_dim=axes_dim, theta=theta)
    return table.to(torch.float16).contiguous()

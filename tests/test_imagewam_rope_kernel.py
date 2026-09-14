"""ImageWAM/FLUX.2 real RoPE correctness (opportunities.md real-math
correction found while starting OPT-002's real-checkpoint work).

Verifies BOTH pieces this project added for real RoPE support:
  1. `flash_rt.models.imagewam.rope.embed_nd_interleaved` (the Python
     precompute of the real 4-axis position table).
  2. `rope_apply_fp16_perhead` (the CUDA kernel that applies it to Q/K).

Against an INDEPENDENT reference transcription of the real upstream
functions (`rope()`, `EmbedND.forward`, `apply_rope()` from
`black-forest-labs/flux2`'s `src/flux2/model.py` at the exact commit
ImageWAM's own `docs/dependencies.md` pins,
`50fe5162777813d869182b139e83b10743caef15` -- fetched and read
directly, not guessed) -- written independently here rather than
imported from `flash_rt/models/imagewam/rope.py` itself, so this is a
genuine two-implementation cross-check of the same real formula, not a
test that imports the code it's testing. Both were ALSO separately
verified once against the real, unmodified upstream file at that exact
commit (not committed to this repo -- see opportunities.md for the
verification note); this test's own independent transcription is what
ships and runs going forward.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.rope import (
    build_backbone_rope_table,
    build_img_ids,
    build_txt_ids,
    embed_nd_interleaved,
)

DEV = "cuda"


def _ref_rope_freqs(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Independent transcription of the real `rope()`."""
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("n,d->nd", pos.float(), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    # real stack order [cos, -sin, sin, cos] reshaped to (..., 2, 2)
    row0 = torch.stack([cos, -sin], dim=-1)   # (seq, dim/2, 2)
    row1 = torch.stack([sin, cos], dim=-1)    # (seq, dim/2, 2)
    return torch.stack([row0, row1], dim=-2)  # (seq, dim/2, 2, 2)


def _ref_embed_nd(ids: torch.Tensor, axes_dim, theta: int) -> torch.Tensor:
    """Independent transcription of the real `EmbedND.forward`."""
    parts = [_ref_rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return torch.cat(parts, dim=-3)  # (seq, sum(axes_dim)/2, 2, 2)


def _ref_apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Independent transcription of the real `apply_rope` (single-tensor
    half -- the real function does Q and K together, math is identical
    per tensor). x: (seq, NH, HD). freqs: (seq, HD/2, 2, 2)."""
    seq, NH, HD = x.shape
    x_ = x.float().reshape(seq, NH, HD // 2, 1, 2)
    f = freqs.unsqueeze(1)  # (seq, 1, HD/2, 2, 2) -- broadcast over NH
    out0 = f[..., 0, 0] * x_[..., 0, 0] + f[..., 0, 1] * x_[..., 0, 1]
    out1 = f[..., 1, 0] * x_[..., 0, 0] + f[..., 1, 1] * x_[..., 0, 1]
    return torch.stack([out0, out1], dim=-1).reshape(seq, NH, HD)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


AXES_DIM = (32, 32, 32, 32)
THETA = 2000


def test_embed_nd_interleaved_matches_independent_reference():
    x0, ref_h, ref_w = 5, 3, 4
    txt_ids = build_txt_ids(x0, device=DEV)
    img_ids = build_img_ids(ref_h, ref_w, 10.0, device=DEV)
    ids = torch.cat([txt_ids, img_ids], dim=0)

    my_table = embed_nd_interleaved(ids, axes_dim=AXES_DIM, theta=THETA)  # (seq, 128)
    ref_freqs = _ref_embed_nd(ids, AXES_DIM, THETA)  # (seq, 64, 2, 2)

    my_cos = my_table[:, 0::2]
    my_sin = my_table[:, 1::2]
    ref_cos = ref_freqs[:, :, 0, 0]
    ref_sin = ref_freqs[:, :, 1, 0]

    cos_diff = (my_cos - ref_cos).abs().max().item()
    sin_diff = (my_sin - ref_sin).abs().max().item()
    print(f"embed_nd_interleaved vs independent reference: cos_diff={cos_diff:.2e} sin_diff={sin_diff:.2e}")
    assert cos_diff < 1e-5
    assert sin_diff < 1e-5


def test_rope_apply_fp16_perhead_matches_independent_reference():
    torch.manual_seed(0)
    x0, ref_h, ref_w = 5, 3, 4
    NH, HD = 4, 128
    seq = x0 + ref_h * ref_w

    table = build_backbone_rope_table(x0, ref_h, ref_w, device=DEV)  # (seq, 128) fp16
    txt_ids = build_txt_ids(x0, device=DEV)
    img_ids = build_img_ids(ref_h, ref_w, 10.0, device=DEV)
    ids = torch.cat([txt_ids, img_ids], dim=0)
    ref_freqs = _ref_embed_nd(ids, AXES_DIM, THETA)

    Q = torch.randn(seq, NH, HD, dtype=torch.float16, device=DEV)
    K = torch.randn(seq, NH, HD, dtype=torch.float16, device=DEV)
    Q_ref = _ref_apply_rope(Q, ref_freqs)
    K_ref = _ref_apply_rope(K, ref_freqs)

    Q_kernel = Q.clone()
    K_kernel = K.clone()
    fvk.rope_apply_fp16_perhead(Q_kernel.data_ptr(), table.data_ptr(), seq, NH, HD, 0)
    fvk.rope_apply_fp16_perhead(K_kernel.data_ptr(), table.data_ptr(), seq, NH, HD, 0)
    torch.cuda.synchronize()

    cos_q = _cosine(Q_ref, Q_kernel)
    cos_k = _cosine(K_ref, K_kernel)
    print(f"rope_apply_fp16_perhead vs independent reference: Q cosine={cos_q:.6f} K cosine={cos_k:.6f}")
    assert cos_q > 0.999
    assert cos_k > 0.999


def test_rope_apply_real_backbone_dims():
    """Real ImageWAM action-inference dims: x0=128 text tokens, a
    plausible ref-image latent grid (24x32=768, matching this project's
    already-established A0-X0=768 image-token span -- see OPT-008)."""
    torch.manual_seed(1)
    x0, ref_h, ref_w = 128, 24, 32
    NH, HD = 24, 128
    seq = x0 + ref_h * ref_w

    table = build_backbone_rope_table(x0, ref_h, ref_w, device=DEV)
    txt_ids = build_txt_ids(x0, device=DEV)
    img_ids = build_img_ids(ref_h, ref_w, 10.0, device=DEV)
    ids = torch.cat([txt_ids, img_ids], dim=0)
    ref_freqs = _ref_embed_nd(ids, AXES_DIM, THETA)

    Q = torch.randn(seq, NH, HD, dtype=torch.float16, device=DEV)
    Q_ref = _ref_apply_rope(Q, ref_freqs)
    Q_kernel = Q.clone()
    fvk.rope_apply_fp16_perhead(Q_kernel.data_ptr(), table.data_ptr(), seq, NH, HD, 0)
    torch.cuda.synchronize()

    cos = _cosine(Q_ref, Q_kernel)
    print(f"real backbone dims: cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_embed_nd_interleaved_matches_independent_reference()
    test_rope_apply_fp16_perhead_matches_independent_reference()
    test_rope_apply_real_backbone_dims()
    print("PASS")

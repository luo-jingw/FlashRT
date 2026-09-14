"""ImageWAM/FLUX.2 real AdaLN modulation correctness (opportunities.md,
found alongside RoPE/QK-Norm/the real mask while starting OPT-002's
real-checkpoint work).

Verifies `flash_rt.models.imagewam.adaln`'s functions against an
INDEPENDENT transcription of the real `timestep_embedding`/
`MLPEmbedder`/`Modulation`/`LayerNorm` composition from
`black-forest-labs/flux2`'s `src/flux2/model.py` (pinned commit
`50fe5162777813d869182b139e83b10743caef15`, fetched and read directly),
written separately here rather than imported from the module under
test -- same two-implementation cross-check discipline used throughout
this project's real-math verification work.
"""
import math

import torch
import torch.nn.functional as F

from flash_rt.models.imagewam.adaln import (
    apply_gated_residual,
    apply_modulation,
    layer_norm_no_affine_fp16,
    mlp_embedder,
    modulation,
    timestep_embedding_real,
)

DEV = "cuda"
FP16 = torch.float16


def _ref_timestep_embedding(t, dim=256, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _ref_mlp_embedder(x, w_in, w_out):
    return F.linear(F.silu(F.linear(x, w_in)), w_out)


def _ref_modulation(vec, w_lin, double):
    multiplier = 6 if double else 3
    out = F.linear(F.silu(vec), w_lin)
    if out.ndim == 2:
        out = out[:, None, :]
    chunks = out.chunk(multiplier, dim=-1)
    return (chunks[:3], chunks[3:]) if double else (chunks[:3], None)


def _ref_layer_norm_no_affine(x, eps=1e-6):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    return ((xf - mean) * torch.rsqrt(var + eps)).to(x.dtype)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_timestep_embedding_matches_reference():
    torch.manual_seed(0)
    t = torch.rand(1, device=DEV)
    mine = timestep_embedding_real(t)
    ref = _ref_timestep_embedding(t)
    diff = (mine.float() - ref.float()).abs().max().item()
    print(f"timestep_embedding max abs diff: {diff:.2e}")
    assert diff < 1e-4


def test_mlp_embedder_matches_reference():
    torch.manual_seed(1)
    hidden = 3072
    x = torch.randn(1, 256, dtype=torch.float32, device=DEV)
    w_in = torch.randn(hidden, 256, dtype=torch.float32, device=DEV) * 0.02
    w_out = torch.randn(hidden, hidden, dtype=torch.float32, device=DEV) * 0.02

    mine = mlp_embedder(x, w_in, w_out)
    ref = _ref_mlp_embedder(x, w_in, w_out)
    cos = _cosine(mine, ref)
    print(f"mlp_embedder cosine: {cos:.6f}")
    assert cos > 0.9999


def test_modulation_matches_reference():
    torch.manual_seed(2)
    hidden = 3072
    vec = torch.randn(1, hidden, dtype=torch.float32, device=DEV)
    w_lin = torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02

    mine1, mine2 = modulation(vec, w_lin, double=True)
    ref1, ref2 = _ref_modulation(vec, w_lin, double=True)
    for a, b in zip(mine1 + mine2, ref1 + ref2):
        cos = _cosine(a, b)
        assert cos > 0.9999
    print("modulation (double) all 6 chunks match reference")


def test_layer_norm_no_affine_matches_reference():
    torch.manual_seed(3)
    seq, dim = 896, 3072
    x = torch.randn(seq, dim, dtype=FP16, device=DEV)
    mine = layer_norm_no_affine_fp16(x)
    ref = _ref_layer_norm_no_affine(x)
    cos = _cosine(mine, ref)
    print(f"layer_norm_no_affine_fp16 vs reference (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_full_adaln_pipeline_matches_reference():
    """End to end: timestep -> vec -> modulation -> LayerNorm -> modulate
    -> gated residual, real dims (hidden=3072, seq=896)."""
    torch.manual_seed(4)
    hidden, seq = 3072, 896
    t = torch.rand(1, device=DEV)
    w_time_in = torch.randn(hidden, 256, dtype=torch.float32, device=DEV) * 0.02
    w_time_out = torch.randn(hidden, hidden, dtype=torch.float32, device=DEV) * 0.02
    w_mod = torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02

    x = torch.randn(seq, hidden, dtype=FP16, device=DEV)
    sublayer_out = torch.randn(seq, hidden, dtype=FP16, device=DEV)

    # -- mine --
    emb = timestep_embedding_real(t)
    vec = mlp_embedder(emb, w_time_in, w_time_out)
    (shift1, scale1, gate1), (shift2, scale2, gate2) = modulation(vec, w_mod, double=True)
    x_normed = layer_norm_no_affine_fp16(x)
    x_mod = apply_modulation(x_normed, shift1[0].to(FP16), scale1[0].to(FP16))
    residual = apply_gated_residual(x.float(), sublayer_out.float(), gate1[0].float()).to(FP16)

    # -- independent reference --
    ref_emb = _ref_timestep_embedding(t)
    ref_vec = _ref_mlp_embedder(ref_emb, w_time_in, w_time_out)
    (r_shift1, r_scale1, _), _ = _ref_modulation(ref_vec, w_mod, double=True)
    ref_normed = _ref_layer_norm_no_affine(x)
    ref_mod = (1 + r_scale1[0].to(FP16)) * ref_normed + r_shift1[0].to(FP16)
    ref_residual = (x.float() + gate1[0].float() * sublayer_out.float()).to(FP16)

    cos_mod = _cosine(x_mod, ref_mod)
    cos_res = _cosine(residual, ref_residual)
    print(f"full AdaLN pipeline: modulate cosine={cos_mod:.6f}, gated-residual cosine={cos_res:.6f}")
    assert cos_mod > 0.999
    assert cos_res > 0.9999


if __name__ == "__main__":
    test_timestep_embedding_matches_reference()
    test_mlp_embedder_matches_reference()
    test_modulation_matches_reference()
    test_layer_norm_no_affine_matches_reference()
    test_full_adaln_pipeline_matches_reference()
    print("PASS")

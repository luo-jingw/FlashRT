"""ImageWAM/FLUX.2 real SingleStreamBlock forward correctness
(opportunities.md OPT-002 follow-up).

Compares `flash_rt.models.imagewam.real_single_stream_block.real_single_stream_block_forward_fp16`
against an INDEPENDENT, from-scratch PyTorch reference of the real
`SingleStreamBlock._qkv`/`_out` (`black-forest-labs/flux2`'s
`src/flux2/model.py` at the pinned commit
`50fe5162777813d869182b139e83b10743caef15`, fetched and read directly).
The module's own docstring documents the one deliberate, correctness-
preserving simplification (splitting the real fused `linear1`/`linear2`
into two GEMMs each) -- the reference below computes the SAME split
GEMMs, not the fused form, since they are mathematically identical and
the point of this test is to verify the ORCHESTRATION (order, residual
wiring, buffer layout), which per-piece tests elsewhere already proved
correct in isolation.

**No mask** (corrected from an earlier version that used a "txt sees
all, ref sees only itself" rule): see `real_single_stream_block.py`'s
own docstring for the full correction (ImageWAM's real inference path
always has `target_len=0`, which reduces to full, unmasked visibility).
"""
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.real_single_stream_block import real_single_stream_block_forward_fp16
from flash_rt.models.imagewam.rope import build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
AXES_DIM = (32, 32, 32, 32)
THETA = 2000


def _ref_layer_norm_no_affine(x, eps=1e-6):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + eps)


def _ref_rmsnorm(x, scale, eps=1e-6):
    xf = x.float()
    rrms = torch.rsqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + eps)
    return xf * rrms * scale.float()


def _ref_rope_freqs(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("n,d->nd", pos.float(), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    row0 = torch.stack([cos, -sin], dim=-1)
    row1 = torch.stack([sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _ref_embed_nd(ids, axes_dim, theta):
    parts = [_ref_rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return torch.cat(parts, dim=-3)


def _ref_apply_rope(x, freqs):
    seq, NH, HD = x.shape
    x_ = x.float().reshape(seq, NH, HD // 2, 1, 2)
    f = freqs.unsqueeze(1)
    out0 = f[..., 0, 0] * x_[..., 0, 0] + f[..., 0, 1] * x_[..., 0, 1]
    out1 = f[..., 1, 0] * x_[..., 0, 0] + f[..., 1, 1] * x_[..., 0, 1]
    return torch.stack([out0, out1], dim=-1).reshape(seq, NH, HD)


def _ref_full_attn(Q, K, V, scale):
    q = Q.permute(1, 0, 2).float()
    k = K.permute(1, 0, 2).float()
    v = V.permute(1, 0, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(1, 0, 2).contiguous()


def _ref_single_stream_block(x, w, mod, ids, NH, HD, hidden, mlp_hidden, scale):
    total = x.shape[0]
    shift, scale_mod, gate = mod

    x_normed = _ref_layer_norm_no_affine(x)
    x_mod = (1 + scale_mod[0]) * x_normed + shift[0]

    qkv = F.linear(x_mod, w["qkv"].float())
    q, k, v = [t.reshape(total, NH, HD) for t in qkv.chunk(3, dim=-1)]
    q = _ref_rmsnorm(q, w["query_norm"])
    k = _ref_rmsnorm(k, w["key_norm"])

    freqs = _ref_embed_nd(ids, AXES_DIM, THETA)
    q = _ref_apply_rope(q, freqs)
    k = _ref_apply_rope(k, freqs)

    attn = _ref_full_attn(q.to(FP16), k.to(FP16), v.to(FP16), scale)
    attn_flat = attn.reshape(total, hidden)

    mlp_merged = F.linear(x_mod, w["mlp_in"].float())
    g, u = mlp_merged.chunk(2, dim=-1)
    mlp_gated = F.silu(g) * u

    from_attn = F.linear(attn_flat, w["attn_out"].float())
    from_mlp = F.linear(mlp_gated, w["mlp_out"].float())
    output = from_attn + from_mlp

    return (x.float() + gate[0] * output).to(FP16)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _make_weights(hidden, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    return {
        "qkv": lin(3 * hidden, hidden),
        "mlp_in": lin(mlp_hidden * 2, hidden),
        "attn_out": lin(hidden, hidden),
        "mlp_out": lin(hidden, mlp_hidden),
        "query_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
        "key_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
    }


def _make_mod(hidden, device):
    shift = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
    scale = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
    gate = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
    return shift, scale, gate


def _gemm_weights(w):
    return {k: v.t().contiguous() for k, v in w.items() if v.ndim == 2} | \
           {k: v for k, v in w.items() if v.ndim == 1}


def _run_case(x0, img_len, NH, HD, hidden, mlp_hidden, seed):
    torch.manual_seed(seed)
    scale = 1.0 / (HD ** 0.5)
    total = x0 + img_len

    x = torch.randn(total, hidden, dtype=FP16, device=DEV)
    w = _make_weights(hidden, mlp_hidden, HD, DEV)
    mod = _make_mod(hidden, DEV)

    ref_h, ref_w = img_len, 1
    table = build_backbone_rope_table(x0, ref_h, ref_w, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    txt_ids = torch.zeros(x0, 4, dtype=torch.float32, device=DEV)
    txt_ids[:, 3] = torch.arange(x0, dtype=torch.float32, device=DEV)
    img_ids = torch.zeros(ref_h, ref_w, 4, dtype=torch.float32, device=DEV)
    img_ids[..., 0] = 10.0
    img_ids[..., 1] = torch.arange(ref_h, dtype=torch.float32, device=DEV)[:, None]
    img_ids[..., 2] = torch.arange(ref_w, dtype=torch.float32, device=DEV)[None, :]
    ids = torch.cat([txt_ids, img_ids.reshape(ref_h * ref_w, 4)], dim=0)

    ref = _ref_single_stream_block(x, w, mod, ids, NH, HD, hidden, mlp_hidden, scale)

    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()
    out = real_single_stream_block_forward_fp16(
        gemm, ctx, x.clone(), _gemm_weights(w), mod, table, NH, HD, hidden, mlp_hidden, scale)

    return _cosine(ref, out)


def test_single_stream_block_small_shape():
    cos = _run_case(x0=3, img_len=5, NH=4, HD=128, hidden=512, mlp_hidden=768, seed=0)
    print(f"single_stream_block (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_single_stream_block_real_dims():
    """Real ImageWAM dims: hidden=3072, mlp_hidden=9216, NH=24, HD=128,
    x0=128 text tokens, 768 image tokens (A0-X0)."""
    cos = _run_case(x0=128, img_len=768, NH=24, HD=128, hidden=3072, mlp_hidden=9216, seed=1)
    print(f"single_stream_block (real dims): cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_single_stream_block_small_shape()
    test_single_stream_block_real_dims()
    print("PASS")

"""ImageWAM real ActionDiT (action expert / "mot" site) forward
correctness (opportunities.md OPT-002).

Compares `flash_rt.models.imagewam.real_action_expert`'s double/single
block functions against an INDEPENDENT, from-scratch PyTorch reference
of the real `SlimFlux2DoubleBlock`/`SlimFlux2SingleBlock`
(`action_dit_flux2.py`, ImageWAM's own source) + the real joint
attention orchestration in `mot.py`'s
`forward_flux2_action_with_video_cache` -- action's own fresh Q/K/V
concatenated with a frozen backbone K/V cache, NO mask (see
opportunities.md's real-mask correction: `target_len=0` gives action
full visibility over the whole combined sequence).
"""
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.real_action_expert import (
    real_action_double_block_forward_fp16,
    real_action_single_block_forward_fp16,
)
from flash_rt.models.imagewam.rope import build_action_rope_table

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


def _ref_action_ids(action_len, device):
    ids = torch.zeros(action_len, 4, dtype=torch.float32, device=device)
    ids[:, 0] = 2.0
    ids[:, 1] = torch.arange(action_len, dtype=torch.float32, device=device)
    return ids


def _ref_full_attn_joint(q, k_cat, v_cat, scale):
    """q: (num_action,NH,HD); k_cat/v_cat: (total,NH,HD). No mask."""
    qh = q.permute(1, 0, 2).float()
    kh = k_cat.permute(1, 0, 2).float()
    vh = v_cat.permute(1, 0, 2).float()
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, vh)
    return out.permute(1, 0, 2).contiguous()


def _ref_action_double_block(action, w, mod, cached_k, cached_v, NH, HD, hidden, attn_dim, mlp_hidden, scale):
    (shift1, scale1, gate1), (shift2, scale2, gate2) = mod
    num_action = action.shape[0]

    x_mod1 = (1 + scale1[0]) * _ref_layer_norm_no_affine(action) + shift1[0]
    qkv = F.linear(x_mod1, w["qkv"].float())
    q, k, v = [t.reshape(num_action, NH, HD) for t in qkv.chunk(3, dim=-1)]
    q = _ref_rmsnorm(q, w["query_norm"])
    k = _ref_rmsnorm(k, w["key_norm"])

    ids = _ref_action_ids(num_action, action.device)
    freqs = _ref_embed_nd(ids, AXES_DIM, THETA)
    q = _ref_apply_rope(q, freqs)
    k = _ref_apply_rope(k, freqs)

    k_cat = torch.cat([cached_k.float(), k], dim=0)
    v_cat = torch.cat([cached_v.float(), v], dim=0)
    mixed = _ref_full_attn_joint(q.to(FP16), k_cat.to(FP16), v_cat.to(FP16), scale)
    mixed_flat = mixed.reshape(num_action, attn_dim)

    proj_out = F.linear(mixed_flat, w["proj"].float())
    action = action.float() + gate1[0] * proj_out

    x_mod2 = (1 + scale2[0]) * _ref_layer_norm_no_affine(action.to(FP16)) + shift2[0]
    mlp_merged = F.linear(x_mod2, w["mlp_in"].float())
    g, u = mlp_merged.chunk(2, dim=-1)
    mlp_gated = F.silu(g) * u
    mlp_out = F.linear(mlp_gated, w["mlp_out"].float())
    action = action + gate2[0] * mlp_out
    return action.to(FP16)


def _ref_action_single_block(action, w, mod, cached_k, cached_v, NH, HD, hidden, attn_dim, mlp_hidden, scale):
    shift, scale_mod, gate = mod
    num_action = action.shape[0]

    x_mod = (1 + scale_mod[0]) * _ref_layer_norm_no_affine(action) + shift[0]
    qkv = F.linear(x_mod, w["qkv"].float())
    q, k, v = [t.reshape(num_action, NH, HD) for t in qkv.chunk(3, dim=-1)]
    q = _ref_rmsnorm(q, w["query_norm"])
    k = _ref_rmsnorm(k, w["key_norm"])

    ids = _ref_action_ids(num_action, action.device)
    freqs = _ref_embed_nd(ids, AXES_DIM, THETA)
    q = _ref_apply_rope(q, freqs)
    k = _ref_apply_rope(k, freqs)

    k_cat = torch.cat([cached_k.float(), k], dim=0)
    v_cat = torch.cat([cached_v.float(), v], dim=0)
    mixed = _ref_full_attn_joint(q.to(FP16), k_cat.to(FP16), v_cat.to(FP16), scale)
    mixed_flat = mixed.reshape(num_action, attn_dim)

    mlp_merged = F.linear(x_mod, w["mlp_in"].float())
    g, u = mlp_merged.chunk(2, dim=-1)
    mlp_gated = F.silu(g) * u

    from_attn = F.linear(mixed_flat, w["attn_out"].float())
    from_mlp = F.linear(mlp_gated, w["mlp_out"].float())
    output = from_attn + from_mlp
    return (action.float() + gate[0] * output).to(FP16)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _make_double_weights(hidden, attn_dim, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    return {
        "qkv": lin(3 * attn_dim, hidden),
        "proj": lin(hidden, attn_dim),
        "mlp_in": lin(mlp_hidden * 2, hidden),
        "mlp_out": lin(hidden, mlp_hidden),
        "query_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
        "key_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
    }


def _make_single_weights(hidden, attn_dim, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    return {
        "qkv": lin(3 * attn_dim, hidden),
        "attn_out": lin(hidden, attn_dim),
        "mlp_in": lin(mlp_hidden * 2, hidden),
        "mlp_out": lin(hidden, mlp_hidden),
        "query_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
        "key_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
    }


def _make_mod(hidden, device, double):
    def one_mod():
        shift = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        scale = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        gate = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        return shift, scale, gate
    return (one_mod(), one_mod()) if double else one_mod()


def _gemm_weights(w):
    return {k: v.t().contiguous() for k, v in w.items() if v.ndim == 2} | \
           {k: v for k, v in w.items() if v.ndim == 1}


def _run_double_case(num_action, backbone_total, NH, HD, hidden, mlp_hidden, seed):
    torch.manual_seed(seed)
    attn_dim = NH * HD
    scale = 1.0 / (HD ** 0.5)

    action = torch.randn(num_action, hidden, dtype=FP16, device=DEV) * 0.1
    cached_k = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    cached_v = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    w = _make_double_weights(hidden, attn_dim, mlp_hidden, HD, DEV)
    mod = _make_mod(hidden, DEV, double=True)

    table = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    ref = _ref_action_double_block(action, w, mod, cached_k, cached_v, NH, HD, hidden, attn_dim, mlp_hidden, scale)

    gemm = fvk.GemmRunner()
    out = real_action_double_block_forward_fp16(
        gemm, action.clone(), _gemm_weights(w), mod, table, cached_k, cached_v,
        NH, HD, hidden, mlp_hidden, scale)

    return _cosine(ref, out)


def _run_single_case(num_action, backbone_total, NH, HD, hidden, mlp_hidden, seed):
    torch.manual_seed(seed)
    attn_dim = NH * HD
    scale = 1.0 / (HD ** 0.5)

    action = torch.randn(num_action, hidden, dtype=FP16, device=DEV) * 0.1
    cached_k = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    cached_v = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    w = _make_single_weights(hidden, attn_dim, mlp_hidden, HD, DEV)
    mod = _make_mod(hidden, DEV, double=False)

    table = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    ref = _ref_action_single_block(action, w, mod, cached_k, cached_v, NH, HD, hidden, attn_dim, mlp_hidden, scale)

    gemm = fvk.GemmRunner()
    out = real_action_single_block_forward_fp16(
        gemm, action.clone(), _gemm_weights(w), mod, table, cached_k, cached_v,
        NH, HD, hidden, mlp_hidden, scale)

    return _cosine(ref, out)


def test_action_double_block_small_shape():
    # RoPE's axes_dim=(32,32,32,32) is a fixed real-model property
    # (sums to head_dim=128) -- HD must stay 128 even in a "small"
    # shape test; only seq lengths/NH/hidden/mlp_hidden shrink here.
    cos = _run_double_case(num_action=5, backbone_total=8, NH=4, HD=128, hidden=64, mlp_hidden=96, seed=0)
    print(f"action double block (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_action_double_block_real_dims():
    """Real ActionDiT dims: hidden=1024, attn_dim=NH*HD=3072, mlp_hidden=4096,
    NH=24, HD=128, num_action=64, backbone_total=896."""
    cos = _run_double_case(num_action=64, backbone_total=896, NH=24, HD=128, hidden=1024, mlp_hidden=4096, seed=1)
    print(f"action double block (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_action_single_block_small_shape():
    cos = _run_single_case(num_action=5, backbone_total=8, NH=4, HD=128, hidden=64, mlp_hidden=96, seed=2)
    print(f"action single block (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_action_single_block_real_dims():
    cos = _run_single_case(num_action=64, backbone_total=896, NH=24, HD=128, hidden=1024, mlp_hidden=4096, seed=3)
    print(f"action single block (real dims): cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_action_double_block_small_shape()
    test_action_double_block_real_dims()
    test_action_single_block_small_shape()
    test_action_single_block_real_dims()
    print("PASS")

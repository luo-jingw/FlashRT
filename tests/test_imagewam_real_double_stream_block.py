"""ImageWAM/FLUX.2 real DoubleStreamBlock forward, FULLY COMBINED
(opportunities.md OPT-002 -- the culmination of this round's real-math
work: per-head K/V, RoPE, QK-Norm, the real txt/ref mask, AdaLN
modulation, real LayerNorm, and the real SiLU-gated MLP, all in one
single-layer block forward).

Compares `flash_rt.models.imagewam.real_double_stream_block.real_double_stream_block_forward_fp16`
against an INDEPENDENT, from-scratch PyTorch reference of the real
`DoubleStreamBlock.forward_kv_extract` (`black-forest-labs/flux2`'s
`src/flux2/model.py` at the pinned commit
`50fe5162777813d869182b139e83b10743caef15`, fetched and read directly).
Every individual piece was already verified in its own dedicated test
file; this one exists specifically to catch ordering/interface bugs a
per-piece test can't see (wrong residual order, wrong mod1-vs-mod2
pairing, wrong txt/img concatenation order, etc).
"""
import math

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16
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


def _ref_masked_attn(Q, K, V, scale, x0, total):
    q = Q.permute(1, 0, 2).float()
    k = K.permute(1, 0, 2).float()
    v = V.permute(1, 0, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.ones(total, total, dtype=torch.bool, device=Q.device)
    mask[x0:total, 0:x0] = False
    logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(1, 0, 2).contiguous()


def _ref_mlp(x, w_in, w_out):
    h = F.linear(x.float(), w_in.float())
    g, u = h.chunk(2, dim=-1)
    gated = F.silu(g) * u
    return F.linear(gated, w_out.float())


def _ref_double_stream_block(txt, img, w, mod_txt, mod_img, ids, x0, img_len, NH, HD, hidden, mlp_hidden, scale):
    total = x0 + img_len
    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = mod_txt
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = mod_img

    txt_mod1 = (1 + txt_scale1[0]) * _ref_layer_norm_no_affine(txt) + txt_shift1[0]
    img_mod1 = (1 + img_scale1[0]) * _ref_layer_norm_no_affine(img) + img_shift1[0]

    txt_qkv = F.linear(txt_mod1, w["txt_qkv"].float())
    img_qkv = F.linear(img_mod1, w["img_qkv"].float())
    txt_q, txt_k, txt_v = [t.reshape(x0, NH, HD) for t in txt_qkv.chunk(3, dim=-1)]
    img_q, img_k, img_v = [t.reshape(img_len, NH, HD) for t in img_qkv.chunk(3, dim=-1)]

    txt_q = _ref_rmsnorm(txt_q, w["txt_query_norm"])
    txt_k = _ref_rmsnorm(txt_k, w["txt_key_norm"])
    img_q = _ref_rmsnorm(img_q, w["img_query_norm"])
    img_k = _ref_rmsnorm(img_k, w["img_key_norm"])

    Q = torch.cat([txt_q, img_q], dim=0)
    K = torch.cat([txt_k, img_k], dim=0)
    V = torch.cat([txt_v, img_v], dim=0)

    freqs = _ref_embed_nd(ids, AXES_DIM, THETA)
    Q = _ref_apply_rope(Q, freqs)
    K = _ref_apply_rope(K, freqs)

    attn = _ref_masked_attn(Q.to(FP16), K.to(FP16), V.to(FP16), scale, x0, total)
    attn_flat = attn.reshape(total, hidden)
    txt_attn_out, img_attn_out = attn_flat[:x0], attn_flat[x0:]

    txt_proj = F.linear(txt_attn_out, w["txt_proj"].float())
    img_proj = F.linear(img_attn_out, w["img_proj"].float())

    txt = txt.float() + txt_gate1[0] * txt_proj
    img = img.float() + img_gate1[0] * img_proj

    txt_mod2 = (1 + txt_scale2[0]) * _ref_layer_norm_no_affine(txt.to(FP16)) + txt_shift2[0]
    img_mod2 = (1 + img_scale2[0]) * _ref_layer_norm_no_affine(img.to(FP16)) + img_shift2[0]

    txt_mlp_out = _ref_mlp(txt_mod2, w["txt_mlp_in"], w["txt_mlp_out"])
    img_mlp_out = _ref_mlp(img_mod2, w["img_mlp_in"], w["img_mlp_out"])

    txt = txt + txt_gate2[0] * txt_mlp_out
    img = img + img_gate2[0] * img_mlp_out

    return txt.to(FP16), img.to(FP16)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _make_weights(hidden, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    w = {}
    for side in ("txt", "img"):
        w[f"{side}_qkv"] = lin(3 * hidden, hidden)
        w[f"{side}_proj"] = lin(hidden, hidden)
        w[f"{side}_mlp_in"] = lin(mlp_hidden * 2, hidden)
        w[f"{side}_mlp_out"] = lin(hidden, mlp_hidden)
        w[f"{side}_query_norm"] = (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16)
        w[f"{side}_key_norm"] = (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16)
    return w


def _make_mod(hidden, device):
    def one_mod():
        shift = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        scale = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        gate = torch.randn(1, 1, hidden, dtype=torch.float32, device=device) * 0.1
        return shift, scale, gate
    return one_mod(), one_mod()


def _gemm_weights(w):
    """GEMM (K,N) convention for FlashRT calls -- transpose real (out,in)."""
    return {k: v.t().contiguous() for k, v in w.items() if v.ndim == 2} | \
           {k: v for k, v in w.items() if v.ndim == 1}


def _run_case(x0, img_len, NH, HD, hidden, mlp_hidden, seed):
    torch.manual_seed(seed)
    scale = 1.0 / (HD ** 0.5)
    total = x0 + img_len

    txt = torch.randn(x0, hidden, dtype=FP16, device=DEV)
    img = torch.randn(img_len, hidden, dtype=FP16, device=DEV)
    w = _make_weights(hidden, mlp_hidden, HD, DEV)
    mod_txt = _make_mod(hidden, DEV)
    mod_img = _make_mod(hidden, DEV)

    ref_h = img_len  # square-ish grid isn't required by the mask/rope math
    ref_w = 1
    table = build_backbone_rope_table(x0, ref_h, ref_w, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    txt_ids = torch.zeros(x0, 4, dtype=torch.float32, device=DEV)
    txt_ids[:, 3] = torch.arange(x0, dtype=torch.float32, device=DEV)
    img_ids = torch.zeros(ref_h, ref_w, 4, dtype=torch.float32, device=DEV)
    img_ids[..., 0] = 10.0
    img_ids[..., 1] = torch.arange(ref_h, dtype=torch.float32, device=DEV)[:, None]
    img_ids[..., 2] = torch.arange(ref_w, dtype=torch.float32, device=DEV)[None, :]
    ids = torch.cat([txt_ids, img_ids.reshape(ref_h * ref_w, 4)], dim=0)

    txt_ref, img_ref = _ref_double_stream_block(
        txt, img, w, mod_txt, mod_img, ids, x0, img_len, NH, HD, hidden, mlp_hidden, scale)

    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()
    w_gemm = _gemm_weights(w)
    txt_out, img_out = real_double_stream_block_forward_fp16(
        gemm, ctx, txt.clone(), img.clone(), w_gemm, mod_txt, mod_img, table,
        NH, HD, hidden, mlp_hidden, scale)

    cos_txt = _cosine(txt_ref, txt_out)
    cos_img = _cosine(img_ref, img_out)
    return cos_txt, cos_img


def test_double_stream_block_small_shape():
    cos_txt, cos_img = _run_case(x0=3, img_len=5, NH=4, HD=128, hidden=512, mlp_hidden=768, seed=0)
    print(f"double_stream_block (small): txt cosine={cos_txt:.6f} img cosine={cos_img:.6f}")
    assert cos_txt > 0.999
    assert cos_img > 0.999


def test_double_stream_block_real_dims():
    """Real ImageWAM dims: hidden=3072, mlp_hidden=9216, NH=24, HD=128,
    x0=128 text tokens, 768 image tokens (A0-X0)."""
    cos_txt, cos_img = _run_case(x0=128, img_len=768, NH=24, HD=128, hidden=3072, mlp_hidden=9216, seed=1)
    print(f"double_stream_block (real dims): txt cosine={cos_txt:.6f} img cosine={cos_img:.6f}")
    assert cos_txt > 0.999
    assert cos_img > 0.999


if __name__ == "__main__":
    test_double_stream_block_small_shape()
    test_double_stream_block_real_dims()
    print("PASS")

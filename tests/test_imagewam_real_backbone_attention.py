"""ImageWAM real "backbone" self-attention, COMBINED (opportunities.md).

Each of QK-Norm and RoPE was already verified independently elsewhere
(test_imagewam_qknorm_reuse.py, test_imagewam_rope_kernel.py) -- but a
per-piece test can't catch an ORDERING or INTERFACE bug in how they're
chained together (e.g. RoPE running before QK-Norm, or one step
accidentally reading a stale buffer). This test builds an INDEPENDENT
PyTorch reference that composes the same real formulas in the real
order (QK-Norm -> RoPE -> attention, confirmed from
`DoubleStreamBlock`/`SingleStreamBlock` in the real `flux2/model.py` at
the pinned commit) and compares it against
`flash_rt.models.imagewam.real_backbone_attn.real_backbone_attention_fp16`
end to end.

**No mask**: an earlier version of this test used a "txt sees all, ref
sees only itself" mask. Found while investigating ActionDiT's real
structure that ImageWAM's real inference path (`infer_action_flux2`)
always calls its own mask builder with `target_len=0`, which reduces to
full, unmasked visibility between text and ref -- see
`real_backbone_attn.py`'s own docstring for the full correction.

Still does NOT include AdaLN modulation, LayerNorm, MLP, or residual
connections -- see that module's own docstring for what remains out of
scope.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.real_backbone_attn import real_backbone_attention_fp16
from flash_rt.models.imagewam.rope import build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16


def _ref_rmsnorm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    rrms = torch.rsqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + eps)
    return (xf * rrms).to(dtype=x.dtype) * scale


def _ref_rope_freqs(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("n,d->nd", pos.float(), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    row0 = torch.stack([cos, -sin], dim=-1)
    row1 = torch.stack([sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)  # (seq, dim/2, 2, 2)


def _ref_embed_nd(ids: torch.Tensor, axes_dim, theta: int) -> torch.Tensor:
    parts = [_ref_rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return torch.cat(parts, dim=-3)


def _ref_apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
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


AXES_DIM = (32, 32, 32, 32)
THETA = 2000


def _ref_full_pipeline(Q, K, V, query_norm_scale, key_norm_scale, ids, scale):
    """Independent reference: QK-Norm -> RoPE -> attention, in the real
    order, no mask (see module docstring)."""
    Qn = _ref_rmsnorm(Q, query_norm_scale)
    Kn = _ref_rmsnorm(K, key_norm_scale)
    freqs = _ref_embed_nd(ids, AXES_DIM, THETA)
    Qr = _ref_apply_rope(Qn, freqs).to(Q.dtype)
    Kr = _ref_apply_rope(Kn, freqs).to(K.dtype)
    return _ref_full_attn(Qr, Kr, V, scale)


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _run_flashrt(Q, K, V, query_norm_scale, key_norm_scale, table, total, NH, HD, scale):
    total_pad = total + (total % 2)
    logits = torch.zeros(total * NH, total_pad, dtype=FP16, device=DEV)
    out = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()
    Q_work = Q.clone()
    K_work = K.clone()
    real_backbone_attention_fp16(
        ctx, Q_work.data_ptr(), K_work.data_ptr(), V.data_ptr(),
        query_norm_scale.data_ptr(), key_norm_scale.data_ptr(),
        table.data_ptr(), logits.data_ptr(), out.data_ptr(),
        total, NH, HD, scale, stream=0)
    torch.cuda.synchronize()
    return out


def test_combined_backbone_attention_small_shape():
    # RoPE's axes_dim=(32,32,32,32) is a fixed real-model property
    # (sums to head_dim=128) -- HD must stay 128 even in a "small"
    # shape test; only seq length and NH shrink here.
    torch.manual_seed(0)
    x0, ref_h, ref_w = 3, 2, 3
    total = x0 + ref_h * ref_w
    NH, HD = 4, 128
    scale = 1.0 / (HD ** 0.5)

    Q = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    query_norm_scale = torch.randn(HD, dtype=FP16, device=DEV).abs() + 0.5
    key_norm_scale = torch.randn(HD, dtype=FP16, device=DEV).abs() + 0.5

    table = build_backbone_rope_table(x0, ref_h, ref_w, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    txt_ids = torch.zeros(x0, 4, dtype=torch.float32, device=DEV)
    txt_ids[:, 3] = torch.arange(x0, dtype=torch.float32, device=DEV)
    img_ids = torch.zeros(ref_h, ref_w, 4, dtype=torch.float32, device=DEV)
    img_ids[..., 0] = 10.0
    img_ids[..., 1] = torch.arange(ref_h, dtype=torch.float32, device=DEV)[:, None]
    img_ids[..., 2] = torch.arange(ref_w, dtype=torch.float32, device=DEV)[None, :]
    ids = torch.cat([txt_ids, img_ids.reshape(ref_h * ref_w, 4)], dim=0)

    ref = _ref_full_pipeline(Q, K, V, query_norm_scale, key_norm_scale, ids, scale)
    out = _run_flashrt(Q, K, V, query_norm_scale, key_norm_scale, table, total, NH, HD, scale)

    cos = _cosine(ref, out)
    print(f"combined backbone attention (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_combined_backbone_attention_real_dims():
    torch.manual_seed(1)
    X0, REF_H, REF_W = 128, 24, 32
    TOTAL = X0 + REF_H * REF_W  # 896, matches A0
    NH, HD = 24, 128
    scale = 1.0 / (HD ** 0.5)

    Q = torch.randn(TOTAL, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(TOTAL, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(TOTAL, NH, HD, dtype=FP16, device=DEV)
    query_norm_scale = torch.randn(HD, dtype=FP16, device=DEV).abs() + 0.5
    key_norm_scale = torch.randn(HD, dtype=FP16, device=DEV).abs() + 0.5

    table = build_backbone_rope_table(X0, REF_H, REF_W, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    txt_ids = torch.zeros(X0, 4, dtype=torch.float32, device=DEV)
    txt_ids[:, 3] = torch.arange(X0, dtype=torch.float32, device=DEV)
    img_ids = torch.zeros(REF_H, REF_W, 4, dtype=torch.float32, device=DEV)
    img_ids[..., 0] = 10.0
    img_ids[..., 1] = torch.arange(REF_H, dtype=torch.float32, device=DEV)[:, None]
    img_ids[..., 2] = torch.arange(REF_W, dtype=torch.float32, device=DEV)[None, :]
    ids = torch.cat([txt_ids, img_ids.reshape(REF_H * REF_W, 4)], dim=0)

    ref = _ref_full_pipeline(Q, K, V, query_norm_scale, key_norm_scale, ids, scale)
    out = _run_flashrt(Q, K, V, query_norm_scale, key_norm_scale, table, TOTAL, NH, HD, scale)

    cos = _cosine(ref, out)
    print(f"combined backbone attention (real dims): cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_combined_backbone_attention_small_shape()
    test_combined_backbone_attention_real_dims()
    print("PASS")

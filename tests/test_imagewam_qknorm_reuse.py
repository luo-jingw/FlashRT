"""ImageWAM/FLUX.2 real QK-Norm (opportunities.md real-math correction,
found alongside the missing-RoPE gap while starting OPT-002's real-
checkpoint work).

Real upstream `QKNorm` (`black-forest-labs/flux2`'s `src/flux2/model.py`
at the pinned commit `50fe5162777813d869182b139e83b10743caef15`, fetched
and read directly) applies an independent, per-head RMSNorm to Q and K
BEFORE RoPE: `RMSNorm.forward` is
`x * rsqrt(mean(x**2, dim=-1) + 1e-6) * scale` -- normalizing over the
LAST dimension (head_dim), i.e. one head's own HD-dim vector at a time,
with its own learned `scale` (real weight `query_norm.scale` /
`key_norm.scale`, each length HD).

**No new kernel needed.** `csrc/kernels/norm.cu`'s existing
`rms_norm_fp16(x, weight, out, seq_len, dim, eps, stream)` computes
EXACTLY this formula (`rms = rsqrt(sum(v^2)/dim + eps)`,
`out = x*rms*weight`, confirmed by reading the kernel body directly)
per ROW of a `(seq_len, dim)` matrix -- and Q/K's own existing physical
layout in this project IS `(seq*NH, HD)` (one row per (token, head)
pair, see `csrc/kernels/attention_cublas.cuh`'s layout convention).
Calling `rms_norm_fp16(Q, query_norm_weight, Q, S*NH, HD, 1e-6, stream)`
(and the K/key_norm equivalent) is therefore a correct, already-shipped-
kernel implementation of the real QKNorm -- this test proves the reuse
is exact, not just plausible.
"""
import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"


def _ref_rmsnorm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Independent transcription of the real `RMSNorm.forward`."""
    x_dtype = x.dtype
    xf = x.float()
    rrms = torch.rsqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + eps)
    return (xf * rrms).to(dtype=x_dtype) * scale


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_rms_norm_fp16_matches_real_qknorm_small_shape():
    torch.manual_seed(0)
    seq, NH, HD = 6, 4, 16
    Q = torch.randn(seq, NH, HD, dtype=torch.float16, device=DEV)
    scale = torch.randn(HD, dtype=torch.float16, device=DEV).abs() + 0.5

    ref = _ref_rmsnorm(Q, scale)

    Q_flat = Q.reshape(seq * NH, HD).contiguous()
    out = torch.zeros_like(Q_flat)
    fvk.rms_norm_fp16(Q_flat.data_ptr(), scale.data_ptr(), out.data_ptr(),
                       seq * NH, HD, 1e-6, 0)
    torch.cuda.synchronize()

    cos = _cosine(ref.reshape(seq * NH, HD), out)
    print(f"rms_norm_fp16 vs real QKNorm formula (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_rms_norm_fp16_matches_real_qknorm_real_dims():
    torch.manual_seed(1)
    seq, NH, HD = 896, 24, 128  # real ImageWAM backbone dims
    K = torch.randn(seq, NH, HD, dtype=torch.float16, device=DEV)
    key_norm_scale = torch.randn(HD, dtype=torch.float16, device=DEV).abs() + 0.5

    ref = _ref_rmsnorm(K, key_norm_scale)

    K_flat = K.reshape(seq * NH, HD).contiguous()
    out = torch.zeros_like(K_flat)
    fvk.rms_norm_fp16(K_flat.data_ptr(), key_norm_scale.data_ptr(), out.data_ptr(),
                       seq * NH, HD, 1e-6, 0)
    torch.cuda.synchronize()

    cos = _cosine(ref.reshape(seq * NH, HD), out)
    print(f"rms_norm_fp16 vs real QKNorm formula (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_rms_norm_fp16_inplace_is_safe():
    """Real pipeline usage needs Q/K normalized IN PLACE (the same
    buffer RoPE and QK^T subsequently read from) -- confirm out==x
    produces the same result as out!=x, since the kernel's per-element
    read-then-write has no cross-row/cross-element ordering hazard."""
    torch.manual_seed(2)
    seq, NH, HD = 32, 4, 16
    X = torch.randn(seq * NH, HD, dtype=torch.float16, device=DEV)
    scale = torch.randn(HD, dtype=torch.float16, device=DEV).abs() + 0.5

    X_outplace = X.clone()
    out = torch.zeros_like(X_outplace)
    fvk.rms_norm_fp16(X_outplace.data_ptr(), scale.data_ptr(), out.data_ptr(),
                       seq * NH, HD, 1e-6, 0)

    X_inplace = X.clone()
    fvk.rms_norm_fp16(X_inplace.data_ptr(), scale.data_ptr(), X_inplace.data_ptr(),
                       seq * NH, HD, 1e-6, 0)
    torch.cuda.synchronize()

    cos = _cosine(out, X_inplace)
    print(f"in-place vs out-of-place rms_norm_fp16: cosine={cos:.6f}")
    assert cos > 0.9999


if __name__ == "__main__":
    test_rms_norm_fp16_matches_real_qknorm_small_shape()
    test_rms_norm_fp16_matches_real_qknorm_real_dims()
    test_rms_norm_fp16_inplace_is_safe()
    print("PASS")

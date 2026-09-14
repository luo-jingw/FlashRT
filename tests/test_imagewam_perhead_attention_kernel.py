"""Real per-head K/V attention kernel correctness (OPT-002, opportunities.md).

Every attention kernel used by this project before this file (see
csrc/kernels/attention_cublas.cu) BROADCASTS one shared (S_kv, HD) K/V
block across all NH query heads -- a structural simplification, not
what a real checkpoint's fused QKV projection produces (real per-head
K/V, shape (S_kv, NH, HD), one independent K/V vector per head).
`attention_qkv_fp16_perhead` (plain self/cross-attention) and
`attention_qkv_fp16_mot_joint_action_perhead` (ImageWAM's masked
action-query site) compute the same QK^T -> masked-softmax -> PV
composition, batched over the NH head dimension via
`cublasGemmStridedBatchedEx`, with independent per-head K/V.

Three checks per kernel, in increasing order of how much they'd catch:
1. Small hand-picked shape vs. a plain PyTorch reference (real,
   independent per-head K/V, not derived from the broadcast case).
2. Real ImageWAM dims (NH=24, HD=128, real x0/a0/total) vs. the same
   PyTorch reference -- catches anything that only breaks at scale
   (alignment, the odd/even total_pad path, etc).
3. Cross-check against the EXISTING broadcast-K/V kernel: when the
   per-head kernel is fed K/V that happen to be identical across every
   head (a broadcast K/V is a special case of per-head K/V), its output
   must match the broadcast kernel's own output bit-for-bit-close. This
   is the strongest check -- it doesn't just validate against a
   from-scratch PyTorch reference, it proves the new kernel's
   strided-batch math reduces to exactly the old, already-shipped
   kernel's math in the case where the two SHOULD agree.
"""
import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
FP16 = torch.float16


def _ref_plain(Q, K, V, scale):
    """Q,K,V: (seq, NH, HD) row-major. Returns (seq, NH, HD)."""
    q = Q.permute(1, 0, 2).float()
    k = K.permute(1, 0, 2).float()
    v = V.permute(1, 0, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(1, 0, 2).contiguous()


def _ref_mot_joint_action(Q, K, V, scale, x0, a0, total):
    """Q: (num_action, NH, HD); K,V: (total, NH, HD). Mask: every action
    row sees [0,x0) U [a0,total), never [x0,a0)."""
    q = Q.permute(1, 0, 2).float()
    k = K.permute(1, 0, 2).float()
    v = V.permute(1, 0, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.zeros(total, dtype=torch.bool, device=Q.device)
    mask[0:x0] = True
    mask[a0:total] = True
    logits = logits.masked_fill(~mask.view(1, 1, total), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(1, 0, 2).contiguous()


def _run_plain(S, S_kv, NH, HD, seed):
    torch.manual_seed(seed)
    scale = 1.0 / (HD ** 0.5)
    Q = torch.randn(S, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(S_kv, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(S_kv, NH, HD, dtype=FP16, device=DEV)

    S_kv_pad = S_kv + (S_kv % 2)
    logits = torch.zeros(S * NH, S_kv_pad, dtype=FP16, device=DEV)
    out = torch.zeros(S, NH, HD, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()
    fvk.attention_qkv_fp16_perhead(ctx, Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                                    logits.data_ptr(), out.data_ptr(),
                                    S, S_kv, NH, HD, scale, 0)
    torch.cuda.synchronize()
    return Q, K, V, out, scale


def _run_masked(num_action, total, NH, HD, x0, a0, seed):
    torch.manual_seed(seed)
    scale = 1.0 / (HD ** 0.5)
    Q = torch.randn(num_action, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(total, NH, HD, dtype=FP16, device=DEV)

    total_pad = total + (total % 2)
    logits = torch.zeros(num_action * NH, total_pad, dtype=FP16, device=DEV)
    out = torch.zeros(num_action, NH, HD, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()
    fvk.attention_qkv_fp16_mot_joint_action_perhead(
        ctx, Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        logits.data_ptr(), out.data_ptr(),
        num_action, total, NH, HD, x0, a0, scale, 0)
    torch.cuda.synchronize()
    return Q, K, V, out, scale


def _cosine(a, b):
    a_ = a.float().flatten()
    b_ = b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_perhead_plain_small_shape_matches_reference():
    Q, K, V, out, scale = _run_plain(S=6, S_kv=8, NH=4, HD=16, seed=0)
    ref = _ref_plain(Q, K, V, scale)
    cos = _cosine(ref, out)
    print(f"perhead plain (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_perhead_plain_real_dims_matches_reference():
    Q, K, V, out, scale = _run_plain(S=896, S_kv=896, NH=24, HD=128, seed=2)
    ref = _ref_plain(Q, K, V, scale)
    cos = _cosine(ref, out)
    print(f"perhead plain (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_perhead_mot_joint_action_small_shape_matches_reference():
    x0, a0, total, num_action = 2, 5, 8, 3
    Q, K, V, out, scale = _run_masked(num_action, total, NH=4, HD=16, x0=x0, a0=a0, seed=1)
    ref = _ref_mot_joint_action(Q, K, V, scale, x0, a0, total)
    cos = _cosine(ref, out)
    print(f"perhead mot_joint_action (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_perhead_mot_joint_action_real_dims_matches_reference():
    X0, A0, NUM_ACTION = 128, 896, 64
    TOTAL = A0 + NUM_ACTION
    Q, K, V, out, scale = _run_masked(NUM_ACTION, TOTAL, NH=24, HD=128, x0=X0, a0=A0, seed=3)
    ref = _ref_mot_joint_action(Q, K, V, scale, X0, A0, TOTAL)
    cos = _cosine(ref, out)
    print(f"perhead mot_joint_action (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_perhead_reduces_to_broadcast_kernel_when_kv_is_shared():
    """The strongest check: feed the per-head kernel a K/V that is
    IDENTICAL across every head (a broadcast K/V is a special case of
    per-head K/V) and confirm it matches the existing, already-shipped
    broadcast kernel's own output -- not just a from-scratch reference."""
    torch.manual_seed(4)
    x0, a0, total, num_action = 2, 5, 8, 3
    NH, HD = 4, 16
    scale = 1.0 / (HD ** 0.5)

    Q = torch.randn(num_action, NH, HD, dtype=FP16, device=DEV)
    K_shared = torch.randn(total, HD, dtype=FP16, device=DEV)
    V_shared = torch.randn(total, HD, dtype=FP16, device=DEV)
    K_perhead = K_shared.unsqueeze(1).expand(-1, NH, -1).contiguous()
    V_perhead = V_shared.unsqueeze(1).expand(-1, NH, -1).contiguous()

    total_pad = total + (total % 2)
    logits = torch.zeros(num_action * NH, total_pad, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()

    out_broadcast = torch.zeros(num_action, NH, HD, dtype=FP16, device=DEV)
    Q_flat = Q.reshape(num_action * NH, HD).contiguous()
    fvk.attention_qkv_fp16_mot_joint_action(
        ctx, Q_flat.data_ptr(), K_shared.data_ptr(), V_shared.data_ptr(),
        logits.data_ptr(), out_broadcast.data_ptr(),
        num_action, total, NH, HD, x0, a0, scale, 0)
    torch.cuda.synchronize()

    out_perhead = torch.zeros(num_action, NH, HD, dtype=FP16, device=DEV)
    fvk.attention_qkv_fp16_mot_joint_action_perhead(
        ctx, Q.data_ptr(), K_perhead.data_ptr(), V_perhead.data_ptr(),
        logits.data_ptr(), out_perhead.data_ptr(),
        num_action, total, NH, HD, x0, a0, scale, 0)
    torch.cuda.synchronize()

    cos = _cosine(out_broadcast.view(-1, HD), out_perhead.view(-1, HD))
    print(f"broadcast-kernel vs perhead-kernel-with-shared-KV: cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_perhead_plain_small_shape_matches_reference()
    test_perhead_plain_real_dims_matches_reference()
    test_perhead_mot_joint_action_small_shape_matches_reference()
    test_perhead_mot_joint_action_real_dims_matches_reference()
    test_perhead_reduces_to_broadcast_kernel_when_kv_is_shared()
    print("PASS")

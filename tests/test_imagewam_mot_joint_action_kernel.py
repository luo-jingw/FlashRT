"""ImageWAM mot_joint_action attention kernel correctness (OPT-003 fix).

Same mask rule as test_imagewam_mot_joint_kernel.py's mot_joint kernel,
restricted to action queries only (opportunities.md OPT-003: the
denoise loop never has a live query on the prefix/image rows, so
computing attention for them is pure waste -- confirmed on real Thor
hardware to leave the denoise step's cost flat across every precision
tested). This kernel takes only the action rows' Q; K/V still cover
the whole combined sequence.

Mask rule (identical to the full mot_joint kernel, restricted to
action rows): action row sees [0, x0) U [a0, total) -- never
[x0, a0).
"""
import torch
import torch.nn.functional as F

import flash_rt_kernels as fvk


def _reference(q_action, k, v, x0, a0, total, num_action, NH, HD, scale):
    # q_action: (num_action*NH, HD); k,v: (total, HD) single-head broadcast.
    q4 = q_action.view(num_action, NH, HD).permute(1, 0, 2)  # (NH, num_action, HD)
    k4 = k.view(total, 1, HD).permute(1, 0, 2).expand(NH, total, HD)
    v4 = v.view(total, 1, HD).permute(1, 0, 2).expand(NH, total, HD)

    mask = torch.zeros(num_action, total, dtype=torch.bool)
    mask[:, :x0] = True
    mask[:, a0:total] = True

    out = F.scaled_dot_product_attention(
        q4.unsqueeze(0).float(), k4.unsqueeze(0).float(), v4.unsqueeze(0).float(),
        attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=scale,
    )  # (1, NH, num_action, HD)
    return out.squeeze(0).permute(1, 0, 2).reshape(num_action * NH, HD)


def test_mot_joint_action_matches_reference():
    torch.manual_seed(0)
    NH, HD = 24, 128
    x0, a0, total = 8, 16, 24  # 8 prefix + 8 target-image + 8 action tokens
    num_action = total - a0
    scale = 1.0 / (HD ** 0.5)

    device = "cuda"
    q = torch.randn(num_action * NH, HD, dtype=torch.float16, device=device)
    k = torch.randn(total, HD, dtype=torch.float16, device=device)
    v = torch.randn(total, HD, dtype=torch.float16, device=device)

    ref = _reference(q.cpu(), k.cpu(), v.cpu(), x0, a0, total, num_action, NH, HD, scale).to(device).half()

    total_pad = total + (total % 2)
    logits = torch.zeros(num_action * NH, total_pad, dtype=torch.float16, device=device)
    out = torch.zeros(num_action * NH, HD, dtype=torch.float16, device=device)

    ctx = fvk.FvkContext()
    fvk.attention_qkv_fp16_mot_joint_action(
        ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
        logits.data_ptr(), out.data_ptr(),
        num_action, total, NH, HD, x0, a0, scale, 0,
    )
    torch.cuda.synchronize()

    a_ = ref.float().flatten()
    b_ = out.float().flatten()
    cos = torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)
    rel_l2 = (b_ - a_).norm() / (a_.norm() + 1e-12)
    print(f"cosine={cos.item():.6f} rel_l2={rel_l2.item():.6f}")
    assert cos.item() > 0.999, f"cosine too low: {cos.item()}"


def test_mot_joint_action_matches_full_mot_joint_on_action_rows():
    """The action rows' own output must be numerically identical whether
    computed via the full mot_joint kernel (over the whole sequence,
    wasteful) or the action-only kernel (OPT-003's fix) -- this is the
    actual correctness contract the optimization must not break."""
    torch.manual_seed(1)
    NH, HD = 24, 128
    x0, a0, total = 8, 16, 24
    num_action = total - a0
    scale = 1.0 / (HD ** 0.5)
    device = "cuda"

    q_full = torch.randn(total * NH, HD, dtype=torch.float16, device=device)
    k = torch.randn(total, HD, dtype=torch.float16, device=device)
    v = torch.randn(total, HD, dtype=torch.float16, device=device)

    ctx = fvk.FvkContext()

    total_pad = total + (total % 2)
    logits_full = torch.zeros(total * NH, total_pad, dtype=torch.float16, device=device)
    out_full = torch.zeros(total * NH, HD, dtype=torch.float16, device=device)
    fvk.attention_qkv_fp16_mot_joint(
        ctx, q_full.data_ptr(), k.data_ptr(), v.data_ptr(),
        logits_full.data_ptr(), out_full.data_ptr(),
        total, NH, HD, x0, a0, scale, 0,
    )
    torch.cuda.synchronize()

    # Action rows of q_full, viewed the same (total, NH, HD) -> (num_action, NH, HD) way.
    q_action = q_full.view(total, NH, HD)[a0:total].reshape(num_action * NH, HD).contiguous()
    logits_action = torch.zeros(num_action * NH, total_pad, dtype=torch.float16, device=device)
    out_action = torch.zeros(num_action * NH, HD, dtype=torch.float16, device=device)
    fvk.attention_qkv_fp16_mot_joint_action(
        ctx, q_action.data_ptr(), k.data_ptr(), v.data_ptr(),
        logits_action.data_ptr(), out_action.data_ptr(),
        num_action, total, NH, HD, x0, a0, scale, 0,
    )
    torch.cuda.synchronize()

    out_full_action_rows = out_full.view(total, NH, HD)[a0:total].reshape(num_action * NH, HD)
    a_ = out_full_action_rows.float().flatten()
    b_ = out_action.float().flatten()
    cos = torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)
    print(f"full-vs-action-only cosine on action rows={cos.item():.6f}")
    assert cos.item() > 0.9999


if __name__ == "__main__":
    test_mot_joint_action_matches_reference()
    test_mot_joint_action_matches_full_mot_joint_on_action_rows()
    print("PASS")

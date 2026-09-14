"""ImageWAM mot_joint attention kernel correctness (plan.md Phase 2).

Compares attention_qkv_fp16_mot_joint against a plain-PyTorch
scaled_dot_product_attention reference with an explicit boolean mask,
at ImageWAM's real per-head geometry (NH=24, HD=128). Checks the
kernel against its own mathematical definition -- not a calibration or
accuracy check against a trained model, and not a Thor-specific
performance measurement (built and run on this project's own Ada
sm_89 GPU; see PROJECT.md for the build fix pybind11 needed).

Mask rule (see csrc/kernels/attention_cublas.cuh / softmax.cuh):
  prefix       [0, x0)      sees [0, x0)
  target-image [x0, a0)     sees [0, a0)
  action       [a0, total)  sees [0, x0) U [a0, total)
"""
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk


def _reference(q, k, v, x0, a0, total, NH, HD, scale):
    # q,k,v: (total*NH, HD) row-major, matching the kernel's own layout.
    q4 = q.view(total, NH, HD).permute(1, 0, 2)  # (NH, total, HD)
    k4 = k.view(total, 1, HD).permute(1, 0, 2).expand(NH, total, HD)
    v4 = v.view(total, 1, HD).permute(1, 0, 2).expand(NH, total, HD)

    mask = torch.zeros(total, total, dtype=torch.bool)
    for row in range(total):
        if row < x0:
            mask[row, :x0] = True
        elif row < a0:
            mask[row, :a0] = True
        else:
            mask[row, :x0] = True
            mask[row, a0:total] = True

    out = F.scaled_dot_product_attention(
        q4.unsqueeze(0).float(), k4.unsqueeze(0).float(), v4.unsqueeze(0).float(),
        attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=scale,
    )  # (1, NH, total, HD)
    return out.squeeze(0).permute(1, 0, 2).reshape(total * NH, HD)


def test_mot_joint_matches_reference():
    torch.manual_seed(0)
    NH, HD = 24, 128
    x0, a0, total = 8, 16, 24  # 8 prefix + 8 target-image + 8 action tokens
    scale = 1.0 / (HD ** 0.5)

    device = "cuda"
    q = torch.randn(total * NH, HD, dtype=torch.float16, device=device)
    k = torch.randn(total, HD, dtype=torch.float16, device=device)
    v = torch.randn(total, HD, dtype=torch.float16, device=device)

    ref = _reference(q.cpu(), k.cpu(), v.cpu(), x0, a0, total, NH, HD, scale).to(device).half()

    total_pad = total + (total % 2)
    logits = torch.zeros(total * NH, total_pad, dtype=torch.float16, device=device)
    out = torch.zeros(total * NH, HD, dtype=torch.float16, device=device)

    ctx = fvk.FvkContext()
    fvk.attention_qkv_fp16_mot_joint(
        ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
        logits.data_ptr(), out.data_ptr(),
        total, NH, HD, x0, a0, scale, 0,
    )
    torch.cuda.synchronize()

    a = ref.float().flatten()
    b = out.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    rel_l2 = (b - a).norm() / (a.norm() + 1e-12)
    print(f"cosine={cos.item():.6f} rel_l2={rel_l2.item():.6f}")
    assert cos.item() > 0.999, f"cosine too low: {cos.item()}"


if __name__ == "__main__":
    test_mot_joint_matches_reference()
    print("PASS")

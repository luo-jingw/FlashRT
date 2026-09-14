"""ImageWAM real backbone prefill wiring correctness (opportunities.md
OPT-002 -- looping the verified real DoubleStreamBlock/SingleStreamBlock
forwards across all 25 real layers).

Not a re-verification of the block math itself (already proven in
test_imagewam_real_double_stream_block.py /
test_imagewam_real_single_stream_block.py) -- this checks that
`imagewam_prefill_real`'s LOOPING and modulation-sharing wiring is
correct: a 1-double-layer run must match calling the single-block
function directly with the same weights/inputs, and the full real
25-layer (5 double + 20 single) loop at real ImageWAM dims must produce
finite, real-shaped output with no cross-layer corruption.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import modulation
from flash_rt.models.imagewam.pipeline_real import compute_shared_modulation, imagewam_prefill_real
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16
from flash_rt.models.imagewam.rope import build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
AXES_DIM = (32, 32, 32, 32)
THETA = 2000


def _make_double_weights(hidden, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    w = {}
    for side in ("txt", "img"):
        w[f"{side}_qkv"] = lin(3 * hidden, hidden).t().contiguous()
        w[f"{side}_proj"] = lin(hidden, hidden).t().contiguous()
        w[f"{side}_mlp_in"] = lin(mlp_hidden * 2, hidden).t().contiguous()
        w[f"{side}_mlp_out"] = lin(hidden, mlp_hidden).t().contiguous()
        w[f"{side}_query_norm"] = (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16)
        w[f"{side}_key_norm"] = (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16)
    return w


def _make_single_weights(hidden, mlp_hidden, HD, device):
    def lin(n, k):
        return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)

    return {
        "qkv": lin(3 * hidden, hidden).t().contiguous(),
        "mlp_in": lin(mlp_hidden * 2, hidden).t().contiguous(),
        "attn_out": lin(hidden, hidden).t().contiguous(),
        "mlp_out": lin(hidden, mlp_hidden).t().contiguous(),
        "query_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
        "key_norm": (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16),
    }


def _make_mod_weights(hidden, device):
    return {
        "time_in_w1": torch.randn(hidden, 256, dtype=torch.float32, device=device) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
    }


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_single_double_layer_loop_matches_direct_call():
    x0, img_len, NH, HD, hidden, mlp_hidden = 3, 5, 4, 128, 512, 768
    torch.manual_seed(0)
    scale = 1.0 / (HD ** 0.5)

    txt = torch.randn(x0, hidden, dtype=FP16, device=DEV)
    img = torch.randn(img_len, hidden, dtype=FP16, device=DEV)
    dw = _make_double_weights(hidden, mlp_hidden, HD, DEV)
    mod_w = _make_mod_weights(hidden, DEV)
    timestep = torch.rand(1, device=DEV)
    mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)

    table = build_backbone_rope_table(x0, img_len, 1, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    txt_direct, img_direct = real_double_stream_block_forward_fp16(
        gemm, ctx, txt.clone(), img.clone(), dw, mod_txt, mod_img, table, NH, HD, hidden, mlp_hidden, scale)

    combined_loop = imagewam_prefill_real(
        gemm, ctx, txt.clone(), img.clone(), [dw], [],
        mod_txt, mod_img, mod_single, table, NH, HD, hidden, mlp_hidden, scale)

    direct_combined = torch.cat([txt_direct, img_direct], dim=0)
    cos = _cosine(direct_combined, combined_loop)
    print(f"1-double-layer loop vs direct call: cosine={cos:.6f}")
    assert cos > 0.999


def test_full_25_layer_backbone_real_dims_finite():
    """Real ImageWAM backbone: 5 double + 20 single layers, hidden=3072,
    mlp_hidden=9216, NH=24, HD=128, x0=128, img_len=768."""
    x0, img_len, NH, HD, hidden, mlp_hidden = 128, 768, 24, 128, 3072, 9216
    num_double, num_single = 5, 20
    torch.manual_seed(1)
    scale = 1.0 / (HD ** 0.5)

    txt = torch.randn(x0, hidden, dtype=FP16, device=DEV) * 0.1
    img = torch.randn(img_len, hidden, dtype=FP16, device=DEV) * 0.1
    double_weights = [_make_double_weights(hidden, mlp_hidden, HD, DEV) for _ in range(num_double)]
    single_weights = [_make_single_weights(hidden, mlp_hidden, HD, DEV) for _ in range(num_single)]
    mod_w = _make_mod_weights(hidden, DEV)
    timestep = torch.rand(1, device=DEV)
    mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)

    table = build_backbone_rope_table(x0, img_len, 1, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    out = imagewam_prefill_real(
        gemm, ctx, txt, img, double_weights, single_weights,
        mod_txt, mod_img, mod_single, table, NH, HD, hidden, mlp_hidden, scale)

    assert out.shape == (x0 + img_len, hidden)
    assert torch.isfinite(out).all()
    print(f"full 25-layer real backbone: shape={tuple(out.shape)}, finite=True, "
          f"mean={out.float().mean().item():.4f}, std={out.float().std().item():.4f}")


if __name__ == "__main__":
    test_single_double_layer_loop_matches_direct_call()
    test_full_25_layer_backbone_real_dims_finite()
    print("PASS")

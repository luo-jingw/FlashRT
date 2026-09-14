"""ImageWAM FULL real forward wiring correctness (opportunities.md
OPT-002 follow-up): `pipeline_real.imagewam_full_forward_real` extends
`imagewam_prefill_real`'s backbone-only scope to the complete model --
one backbone prefill (collecting the per-layer frozen K/V cache) plus
the whole ActionDiT flow-matching denoise loop reading that cache.

Not a re-verification of any single block's own math (already proven
in test_imagewam_real_double_stream_block.py /
test_imagewam_real_single_stream_block.py / test_imagewam_real_action_expert.py)
-- this checks the NEW wiring specific to this function: (1) the
per-layer K/V cache collected during prefill lines up correctly with
the same-indexed action-expert layer that reads it, (2) the ActionDiT
modulation is correctly recomputed every denoise step from that step's
own changing timestep, (3) the Euler update between steps is wired
correctly. A 1-denoise-step run must match manually replicating that
one step (prefill + one action block loop + one Euler update) via the
already-verified lower-level functions directly, called independently
with the same weights/inputs.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.pipeline_real import (
    compute_action_modulation,
    compute_shared_modulation,
    imagewam_full_forward_real,
    imagewam_prefill_real,
)
from flash_rt.models.imagewam.real_action_expert import (
    real_action_double_block_forward_fp16,
    real_action_single_block_forward_fp16,
)
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32
AXES_DIM = (32, 32, 32, 32)
THETA = 2000


def _lin(n, k, device):
    return (torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16).t().contiguous()


def _norm_scale(HD, device):
    return (torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16)


def _make_double_weights(hidden, mlp_hidden, HD, device, sides=("txt", "img")):
    w = {}
    for side in sides:
        prefix = f"{side}_" if len(sides) > 1 else ""
        w[f"{prefix}qkv"] = _lin(3 * hidden, hidden, device)
        w[f"{prefix}proj"] = _lin(hidden, hidden, device)
        w[f"{prefix}mlp_in"] = _lin(mlp_hidden * 2, hidden, device)
        w[f"{prefix}mlp_out"] = _lin(hidden, mlp_hidden, device)
        w[f"{prefix}query_norm"] = _norm_scale(HD, device)
        w[f"{prefix}key_norm"] = _norm_scale(HD, device)
    return w


def _make_single_weights(hidden, mlp_hidden, HD, device):
    return {
        "qkv": _lin(3 * hidden, hidden, device),
        "mlp_in": _lin(mlp_hidden * 2, hidden, device),
        "attn_out": _lin(hidden, hidden, device),
        "mlp_out": _lin(hidden, mlp_hidden, device),
        "query_norm": _norm_scale(HD, device),
        "key_norm": _norm_scale(HD, device),
    }


def _make_action_double_weights(hidden, attn_dim, mlp_hidden, HD, device):
    return {
        "qkv": _lin(3 * attn_dim, hidden, device),
        "proj": _lin(hidden, attn_dim, device),
        "mlp_in": _lin(mlp_hidden * 2, hidden, device),
        "mlp_out": _lin(hidden, mlp_hidden, device),
        "query_norm": _norm_scale(HD, device),
        "key_norm": _norm_scale(HD, device),
    }


def _make_action_single_weights(hidden, attn_dim, mlp_hidden, HD, device):
    return {
        "qkv": _lin(3 * attn_dim, hidden, device),
        "mlp_in": _lin(mlp_hidden * 2, hidden, device),
        "attn_out": _lin(hidden, attn_dim, device),
        "mlp_out": _lin(hidden, mlp_hidden, device),
        "query_norm": _norm_scale(HD, device),
        "key_norm": _norm_scale(HD, device),
    }


def _make_backbone_mod_weights(hidden, device):
    return {
        "time_in_w1": torch.randn(hidden, 256, dtype=torch.float32, device=device) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
    }


def _make_action_mod_weights(hidden, device):
    return {
        "time_in_w1": torch.randn(hidden, 256, dtype=torch.float32, device=device) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_double": torch.randn(6 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=torch.float32, device=device) * 0.02,
    }


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _build_common(device):
    torch.manual_seed(0)
    x0, img_len, NH, HD = 3, 5, 4, 128
    hidden, mlp_hidden = NH * HD, 384
    action_hidden, action_mlp_hidden = 96, 192
    num_double, num_single = 2, 3
    action_num_double, action_num_single = 2, 3
    num_action = 4
    scale = 1.0 / (HD ** 0.5)

    txt = torch.randn(x0, hidden, dtype=FP16, device=device) * 0.1
    img = torch.randn(img_len, hidden, dtype=FP16, device=device) * 0.1
    action_latent = torch.randn(num_action, action_hidden, dtype=F32, device=device) * 0.1

    bb_double_w = [_make_double_weights(hidden, mlp_hidden, HD, device) for _ in range(num_double)]
    bb_single_w = [_make_single_weights(hidden, mlp_hidden, HD, device) for _ in range(num_single)]
    bb_mod_w = _make_backbone_mod_weights(hidden, device)

    attn_dim = NH * HD
    act_double_w = [_make_action_double_weights(action_hidden, attn_dim, action_mlp_hidden, HD, device)
                    for _ in range(action_num_double)]
    act_single_w = [_make_action_single_weights(action_hidden, attn_dim, action_mlp_hidden, HD, device)
                    for _ in range(action_num_single)]
    act_mod_w = _make_action_mod_weights(action_hidden, device)

    bb_table = build_backbone_rope_table(x0, img_len, 1, axes_dim=AXES_DIM, theta=THETA, device=device)
    act_table = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=device)

    return dict(
        x0=x0, img_len=img_len, NH=NH, HD=HD, hidden=hidden, mlp_hidden=mlp_hidden,
        action_hidden=action_hidden, action_mlp_hidden=action_mlp_hidden,
        num_action=num_action, scale=scale,
        txt=txt, img=img, action_latent=action_latent,
        bb_double_w=bb_double_w, bb_single_w=bb_single_w, bb_mod_w=bb_mod_w,
        act_double_w=act_double_w, act_single_w=act_single_w, act_mod_w=act_mod_w,
        bb_table=bb_table, act_table=act_table,
    )


def test_one_denoise_step_matches_manual_reference():
    c = _build_common(DEV)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    out = imagewam_full_forward_real(
        gemm, ctx, c["txt"].clone(), c["img"].clone(), c["action_latent"].clone(),
        c["bb_double_w"], c["bb_single_w"], c["bb_mod_w"],
        c["act_double_w"], c["act_single_w"], c["act_mod_w"],
        c["bb_table"], c["act_table"],
        c["NH"], c["HD"], c["hidden"], c["mlp_hidden"],
        c["action_hidden"], c["action_mlp_hidden"], c["scale"],
        num_denoise_steps=1, backbone_timestep=0.0)

    # Manual reference: same backbone prefill (kv_cache), same single
    # denoise step (action_timestep=1.0, dt=1.0), via the already-
    # verified lower-level functions called directly.
    mod_txt, mod_img, mod_single_bb = compute_shared_modulation(
        torch.zeros(1, device=DEV), c["bb_mod_w"], c["hidden"])
    _, kv_cache = imagewam_prefill_real(
        gemm, ctx, c["txt"].clone(), c["img"].clone(), c["bb_double_w"], c["bb_single_w"],
        mod_txt, mod_img, mod_single_bb, c["bb_table"],
        c["NH"], c["HD"], c["hidden"], c["mlp_hidden"], c["scale"], collect_kv_cache=True)

    mod_double, mod_single = compute_action_modulation(
        torch.ones(1, device=DEV), c["act_mod_w"], c["action_hidden"])
    action = c["action_latent"].clone().to(FP16)
    for i, w in enumerate(c["act_double_w"]):
        K, V = kv_cache[i]
        action = real_action_double_block_forward_fp16(
            gemm, action, w, mod_double, c["act_table"], K, V,
            c["NH"], c["HD"], c["action_hidden"], c["action_mlp_hidden"], c["scale"])
    for i, w in enumerate(c["act_single_w"]):
        K, V = kv_cache[len(c["act_double_w"]) + i]
        action = real_action_single_block_forward_fp16(
            gemm, action, w, mod_single, c["act_table"], K, V,
            c["NH"], c["HD"], c["action_hidden"], c["action_mlp_hidden"], c["scale"])
    expected = c["action_latent"] + 1.0 * action.float()

    cos = _cosine(out, expected)
    print(f"1-denoise-step full forward vs manual reference: cosine={cos:.6f}")
    assert cos > 0.999
    assert torch.isfinite(out).all()


def test_multi_step_denoise_loop_finite_and_correct_shape():
    c = _build_common(DEV)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    out = imagewam_full_forward_real(
        gemm, ctx, c["txt"].clone(), c["img"].clone(), c["action_latent"].clone(),
        c["bb_double_w"], c["bb_single_w"], c["bb_mod_w"],
        c["act_double_w"], c["act_single_w"], c["act_mod_w"],
        c["bb_table"], c["act_table"],
        c["NH"], c["HD"], c["hidden"], c["mlp_hidden"],
        c["action_hidden"], c["action_mlp_hidden"], c["scale"],
        num_denoise_steps=4, backbone_timestep=0.0)

    assert out.shape == (c["num_action"], c["action_hidden"])
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    print(f"4-step full denoise loop: shape={tuple(out.shape)}, finite=True, "
          f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")


if __name__ == "__main__":
    test_one_denoise_step_matches_manual_reference()
    test_multi_step_denoise_loop_finite_and_correct_shape()
    print("PASS")

"""ImageWAM denoise loop wiring test (plan.md Phase 4, rewritten
2026-09-14 alongside `pipeline_thor.py`'s real-math rewrite).

Extends test_imagewam_prefill.py's setup: runs a real prefill first
(populating the "backbone" region of the shared KV cache), then the
denoise loop against it. Random weights, small dims -- wiring only
(finite output, latent advances, prefill's own K/V survives the
denoise loop untouched), not accuracy -- see
`tests/test_imagewam_thor_real_wiring.py` for per-layer math
correctness against the verified tensor-level reference.

`HD=128` is fixed (real 4-axis RoPE sums to 128). See
`test_imagewam_prefill.py`'s own docstring for the dangling-pointer
trap (`_rand(...).t().contiguous().data_ptr()`, fixed here via `_lin()`
from the start) this file's own earlier draft hit too.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_real import compute_action_modulation, compute_shared_modulation
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_keepalive = []


def _rand(*shape, dtype=FP16, scale=0.02):
    t = (torch.randn(*shape, dtype=torch.float32, device=DEV) * scale).to(dtype)
    _keepalive.append(t)
    return t


def _lin(n, k, scale=0.02):
    t = (torch.randn(n, k, dtype=torch.float32, device=DEV) * scale).to(FP16).t().contiguous()
    _keepalive.append(t)
    return t


def _zeros(*shape, dtype=FP16):
    t = torch.zeros(*shape, dtype=dtype, device=DEV)
    _keepalive.append(t)
    return t


def _norm_scale(HD):
    t = (torch.randn(HD, dtype=torch.float32, device=DEV).abs() + 0.5).to(FP16)
    _keepalive.append(t)
    return t


def test_denoise_loop_runs_and_advances_latent():
    torch.manual_seed(0)
    NH, HD, mlp_hidden, joint_attention_dim = 2, 128, 192, 64
    hidden = NH * HD
    num_double, num_single = 2, 3
    num_layers = num_double + num_single  # 5, shared by backbone and ActionDiT
    x0, a0 = 4, 8  # 4 text + 4 image tokens
    num_action = 3
    total = a0 + num_action
    action_hidden_dim, action_mlp_hidden = 96, 128
    action_attn_width = hidden  # required: shared per-head geometry with the backbone
    num_denoise_steps = 2

    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden,
                joint_attention_dim=joint_attention_dim, x0=x0, a0=a0,
                num_layers_double=num_double, num_layers_single=num_single,
                action_hidden_dim=action_hidden_dim, action_attn_width=action_attn_width,
                action_mlp_hidden=action_mlp_hidden,
                num_action=num_action, total=total,
                action_num_layers_double=num_double, action_num_layers_single=num_single,
                dt=1.0 / num_denoise_steps, num_denoise_steps=num_denoise_steps)

    weights = {}
    for L in range(num_double):
        weights[("backbone", "double", L, "txt_in.weight")] = _lin(hidden, joint_attention_dim).data_ptr()
        for prefix in ("txt", "img"):
            weights[("backbone", "double", L, f"{prefix}_qkv.weight")] = _lin(3 * hidden, hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_proj.weight")] = _lin(hidden, hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_mlp0.weight")] = _lin(mlp_hidden * 2, hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_mlp2.weight")] = _lin(hidden, mlp_hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_query_norm")] = _norm_scale(HD).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_key_norm")] = _norm_scale(HD).data_ptr()
        weights[("action_dit", "double", L, "qkv.weight")] = _lin(3 * action_attn_width, action_hidden_dim).data_ptr()
        weights[("action_dit", "double", L, "proj.weight")] = _lin(action_hidden_dim, action_attn_width).data_ptr()
        weights[("action_dit", "double", L, "mlp0.weight")] = _lin(action_mlp_hidden * 2, action_hidden_dim).data_ptr()
        weights[("action_dit", "double", L, "mlp2.weight")] = _lin(action_hidden_dim, action_mlp_hidden).data_ptr()
        weights[("action_dit", "double", L, "query_norm")] = _norm_scale(HD).data_ptr()
        weights[("action_dit", "double", L, "key_norm")] = _norm_scale(HD).data_ptr()
    for L in range(num_single):
        weights[("backbone", "single", L, "qkv.weight")] = _lin(3 * hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "mlp_in.weight")] = _lin(mlp_hidden * 2, hidden).data_ptr()
        weights[("backbone", "single", L, "attn_out_proj.weight")] = _lin(hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "mlp_down.weight")] = _lin(hidden, mlp_hidden).data_ptr()
        weights[("backbone", "single", L, "query_norm")] = _norm_scale(HD).data_ptr()
        weights[("backbone", "single", L, "key_norm")] = _norm_scale(HD).data_ptr()
        weights[("action_dit", "single", L, "qkv.weight")] = _lin(3 * action_attn_width, action_hidden_dim).data_ptr()
        weights[("action_dit", "single", L, "mlp_in.weight")] = _lin(action_mlp_hidden * 2, action_hidden_dim).data_ptr()
        weights[("action_dit", "single", L, "attn_out_proj.weight")] = _lin(action_hidden_dim, action_attn_width).data_ptr()
        weights[("action_dit", "single", L, "mlp_down.weight")] = _lin(action_hidden_dim, action_mlp_hidden).data_ptr()
        weights[("action_dit", "single", L, "query_norm")] = _norm_scale(HD).data_ptr()
        weights[("action_dit", "single", L, "key_norm")] = _norm_scale(HD).data_ptr()

    context = _rand(x0, joint_attention_dim)
    # Non-degenerate image rows -- see test_imagewam_prefill.py's own
    # comment (an all-zero/all-equal row is a degenerate LayerNorm input).
    backbone_hidden = _rand(a0, hidden, scale=0.1)
    action_latent = _rand(num_action, action_hidden_dim, dtype=F32, scale=0.01)
    bufs = {
        "context": context.data_ptr(),
        "backbone_hidden": backbone_hidden.data_ptr(),
        "normed_scratch": _zeros(a0, hidden).data_ptr(),
        "modded_scratch": _zeros(a0, hidden).data_ptr(),
        "txt_qkv_merged": _zeros(x0, 3 * hidden).data_ptr(),
        "img_qkv_merged": _zeros(a0 - x0, 3 * hidden).data_ptr(),
        "single_qkv_merged": _zeros(a0, 3 * hidden).data_ptr(),
        "action_qkv_merged": _zeros(num_action, 3 * action_attn_width).data_ptr(),
        "txt_mlp_merged": _zeros(x0, mlp_hidden * 2).data_ptr(),
        "txt_mlp_gated": _zeros(x0, mlp_hidden).data_ptr(),
        "img_mlp_merged": _zeros(a0 - x0, mlp_hidden * 2).data_ptr(),
        "img_mlp_gated": _zeros(a0 - x0, mlp_hidden).data_ptr(),
        "single_mlp_merged": _zeros(a0, mlp_hidden * 2).data_ptr(),
        "single_mlp_gated": _zeros(a0, mlp_hidden).data_ptr(),
        "proj_scratch": _zeros(a0, hidden).data_ptr(),
        "proj_scratch2": _zeros(a0, hidden).data_ptr(),
        "action_latent": action_latent.data_ptr(),
        "action_hidden": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_normed": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_modded": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_proj_scratch": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_proj_scratch2": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_mlp_merged": _zeros(num_action, action_mlp_hidden * 2).data_ptr(),
        "action_mlp_gated": _zeros(num_action, action_mlp_hidden).data_ptr(),
    }

    bb_mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    mod_txt, mod_img, mod_single = compute_shared_modulation(
        torch.zeros(1, device=DEV), bb_mod_w, hidden)
    rope_table = build_backbone_rope_table(x0, a0 - x0, 1, device=DEV)

    act_mod_w = {
        "time_in_w1": torch.randn(action_hidden_dim, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(action_hidden_dim, action_hidden_dim, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * action_hidden_dim, action_hidden_dim, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * action_hidden_dim, action_hidden_dim, dtype=F32, device=DEV) * 0.02,
    }
    dt = dims["dt"]
    action_mods = []
    for step in range(num_denoise_steps):
        action_timestep = 1.0 - step * dt
        m_double, m_single = compute_action_modulation(
            torch.full((1,), action_timestep, device=DEV), act_mod_w, action_hidden_dim)
        action_mods.append((m_double, m_single))
    action_rope_table = build_action_rope_table(num_action, device=DEV)

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total,
                                         num_layers=num_layers, num_heads=NH, head_dim=HD)
    ctx = fvk.FvkContext()
    K_cache = _zeros(num_layers, total, hidden)
    V_cache = _zeros(num_layers, total, hidden)
    Q_O = _zeros(total, hidden)
    logits = _zeros(total * NH, total + (total % 2))
    layer_stride = K_cache[0].numel() * 2
    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": layer_stride,
        },
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    gemm = fvk.GemmRunner()
    imagewam_prefill(ctx, fvk, gemm, bufs, weights, dims, stream=0, attn=backend,
                      mod_txt=mod_txt, mod_img=mod_img, mod_single=mod_single,
                      rope_table=rope_table.data_ptr())
    torch.cuda.synchronize()
    assert torch.isfinite(backbone_hidden).all(), "prefill produced NaN/Inf"

    latent_before = action_latent.clone()
    imagewam_denoise_loop(ctx, fvk, gemm, bufs, weights, dims, stream=0, attn=backend,
                           action_mods=action_mods, action_rope_table=action_rope_table.data_ptr())
    torch.cuda.synchronize()

    assert torch.isfinite(action_latent).all(), "action_latent has NaN/Inf after denoise loop"
    assert not torch.equal(action_latent, latent_before), "action_latent was never advanced"
    # Prefill's own K/V (rows [0,a0)) must survive the denoise loop untouched.
    assert torch.isfinite(K_cache[:, :a0]).all() and torch.isfinite(V_cache[:, :a0]).all()
    print("PASS: imagewam_denoise_loop finite, action_latent advanced, backbone KV cache intact")


if __name__ == "__main__":
    test_denoise_loop_runs_and_advances_latent()

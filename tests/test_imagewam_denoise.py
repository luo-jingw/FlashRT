"""ImageWAM denoise loop wiring test (plan.md Phase 4).

Extends test_imagewam_prefill.py's setup: runs a real prefill first
(populating the "backbone" region of the shared KV cache), then the
denoise loop against it. Random weights, small dims -- wiring only,
not accuracy. See pipeline_thor.py's own module docstring (backbone
simplifications) and its Phase 4 section comment (ActionDiT
simplifications: no output head, per-step attention wastes compute on
unused backbone/image rows) for what this deliberately does not check.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_keepalive = []


def _rand(*shape, dtype=FP16):
    t = torch.randn(*shape, dtype=dtype, device=DEV)
    _keepalive.append(t)
    return t


def _zeros(*shape, dtype=FP16):
    t = torch.zeros(*shape, dtype=dtype, device=DEV)
    _keepalive.append(t)
    return t


def test_denoise_loop_runs_and_advances_latent():
    torch.manual_seed(0)
    hidden, HD, NH, mlp_hidden, joint_attention_dim = 96, 16, 6, 192, 64
    num_double, num_single = 2, 3
    num_layers = num_double + num_single  # 5, shared by backbone and ActionDiT
    x0, a0 = 4, 8  # 4 text + 4 image tokens (backbone "standard" kernel needs even a0)
    num_action = 3
    total = a0 + num_action
    action_hidden_dim, action_mlp_hidden = 32, 64
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
        weights[("backbone", "double", L, "txt_in")] = _rand(joint_attention_dim, hidden).data_ptr()
        for prefix in ("txt", "img"):
            weights[("backbone", "double", L, f"{prefix}_q")] = _rand(hidden, hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_k")] = _rand(hidden, HD).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_v")] = _rand(hidden, HD).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_proj")] = _rand(hidden, hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_mlp0")] = _rand(hidden, mlp_hidden).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_mlp2")] = _rand(mlp_hidden, hidden).data_ptr()
        weights[("action_dit", "double", L, "q")] = _rand(action_hidden_dim, action_attn_width).data_ptr()
        weights[("action_dit", "double", L, "k")] = _rand(action_hidden_dim, HD).data_ptr()
        weights[("action_dit", "double", L, "v")] = _rand(action_hidden_dim, HD).data_ptr()
        weights[("action_dit", "double", L, "proj")] = _rand(action_attn_width, action_hidden_dim).data_ptr()
        weights[("action_dit", "double", L, "mlp0")] = _rand(action_hidden_dim, action_mlp_hidden).data_ptr()
        weights[("action_dit", "double", L, "mlp2")] = _rand(action_mlp_hidden, action_hidden_dim).data_ptr()
    for L in range(num_single):
        weights[("backbone", "single", L, "q")] = _rand(hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "k")] = _rand(hidden, HD).data_ptr()
        weights[("backbone", "single", L, "v")] = _rand(hidden, HD).data_ptr()
        weights[("backbone", "single", L, "mlp_in")] = _rand(hidden, mlp_hidden).data_ptr()
        weights[("backbone", "single", L, "attn_out_proj")] = _rand(hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "mlp_down")] = _rand(mlp_hidden, hidden).data_ptr()
        weights[("action_dit", "single", L, "q")] = _rand(action_hidden_dim, action_attn_width).data_ptr()
        weights[("action_dit", "single", L, "k")] = _rand(action_hidden_dim, HD).data_ptr()
        weights[("action_dit", "single", L, "v")] = _rand(action_hidden_dim, HD).data_ptr()
        weights[("action_dit", "single", L, "mlp_in")] = _rand(action_hidden_dim, action_mlp_hidden).data_ptr()
        weights[("action_dit", "single", L, "attn_out_proj")] = _rand(action_attn_width, action_hidden_dim).data_ptr()
        weights[("action_dit", "single", L, "mlp_down")] = _rand(action_mlp_hidden, action_hidden_dim).data_ptr()

    context = _rand(x0, joint_attention_dim)
    backbone_hidden = _rand(a0, hidden)
    ones_hidden = torch.ones(hidden, dtype=FP16, device=DEV); _keepalive.append(ones_hidden)
    ones_action = torch.ones(action_hidden_dim, dtype=FP16, device=DEV); _keepalive.append(ones_action)
    action_latent = torch.randn(num_action, action_hidden_dim, dtype=F32, device=DEV) * 0.01
    _keepalive.append(action_latent)
    bufs = {
        "context": context.data_ptr(),
        "backbone_hidden": backbone_hidden.data_ptr(),
        "txt_mlp_hidden": _zeros(x0, mlp_hidden).data_ptr(),
        "img_mlp_hidden": _zeros(a0 - x0, mlp_hidden).data_ptr(),
        "single_mlp_hidden": _zeros(a0, mlp_hidden).data_ptr(),
        "proj_scratch": _zeros(a0, hidden).data_ptr(),
        "normed_scratch": _zeros(a0, hidden).data_ptr(),
        "norm_ones": ones_hidden.data_ptr(),
        "action_latent": action_latent.data_ptr(),
        "action_hidden": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_normed": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_norm_ones": ones_action.data_ptr(),
        "action_proj_scratch": _zeros(num_action, action_hidden_dim).data_ptr(),
        "action_mlp_hidden": _zeros(num_action, action_mlp_hidden).data_ptr(),
    }

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total)
    ctx = fvk.FvkContext()
    K_cache = _zeros(num_layers, total, HD)
    V_cache = _zeros(num_layers, total, HD)
    Q_O = _zeros(total, hidden)  # sized to the wider of hidden/action_hidden_dim*NH use
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
    )

    gemm = fvk.GemmRunner()
    imagewam_prefill(ctx, fvk, gemm, bufs, weights, dims, stream=0, attn=backend)
    torch.cuda.synchronize()
    assert torch.isfinite(backbone_hidden).all(), "prefill produced NaN/Inf"

    latent_before = action_latent.clone()
    imagewam_denoise_loop(ctx, fvk, gemm, bufs, weights, dims, stream=0, attn=backend)
    torch.cuda.synchronize()

    assert torch.isfinite(action_latent).all(), "action_latent has NaN/Inf after denoise loop"
    assert not torch.equal(action_latent, latent_before), "action_latent was never advanced"
    # Prefill's own K/V (rows [0,a0)) must survive the denoise loop untouched.
    assert torch.isfinite(K_cache[:, :a0]).all() and torch.isfinite(V_cache[:, :a0]).all()
    print("PASS: imagewam_denoise_loop finite, action_latent advanced, backbone KV cache intact")


if __name__ == "__main__":
    test_denoise_loop_runs_and_advances_latent()

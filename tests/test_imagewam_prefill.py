"""ImageWAM backbone prefill wiring test (plan.md Phase 3).

Random weights, small dims (not the real FLUX.2-4B size -- this only
checks the pointer-interface wiring: no NaN/Inf, correct shapes, the
KV cache actually gets written). See pipeline_thor.py's own docstring
for the documented simplifications (unweighted RMS norm only, no AdaLN
modulation, single-shared K/V, no persistent text residual) this test
does not re-litigate.

Every buffer/weight tensor is kept alive in `_keepalive` for the whole
test: a bare `torch.zeros(...).data_ptr()` expression drops the only
Python reference to the tensor the instant that line finishes, and
PyTorch's caching allocator is then free to hand that same memory to
the next allocation -- silently corrupting an already-stored pointer
that nothing still references. This is not hypothetical inside a
25-layer loop's worth of allocations.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_thor import imagewam_prefill

DEV = "cuda"
FP16 = torch.float16

_keepalive = []


def _rand(*shape):
    t = torch.randn(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _zeros(*shape):
    t = torch.zeros(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def test_prefill_runs_and_populates_kv_cache():
    torch.manual_seed(0)
    hidden, HD, NH, mlp_hidden, joint_attention_dim = 96, 16, 6, 192, 64
    num_double, num_single = 2, 3
    num_layers = num_double + num_single
    x0, a0 = 4, 8  # 4 text tokens + 4 image tokens
    # a0 (and any other "standard"-kernel kv_seq) must be even -- see
    # ImageWAMAttnBackend.run()'s own comment on attention_qkv_fp16's
    # unpadded __half2 softmax.

    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden,
                joint_attention_dim=joint_attention_dim, x0=x0, a0=a0,
                num_layers_double=num_double, num_layers_single=num_single)

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
    for L in range(num_single):
        weights[("backbone", "single", L, "q")] = _rand(hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "k")] = _rand(hidden, HD).data_ptr()
        weights[("backbone", "single", L, "v")] = _rand(hidden, HD).data_ptr()
        weights[("backbone", "single", L, "mlp_in")] = _rand(hidden, mlp_hidden).data_ptr()
        weights[("backbone", "single", L, "attn_out_proj")] = _rand(hidden, hidden).data_ptr()
        weights[("backbone", "single", L, "mlp_down")] = _rand(mlp_hidden, hidden).data_ptr()

    context = _rand(x0, joint_attention_dim)
    backbone_hidden = _rand(a0, hidden)
    ones = torch.ones(hidden, dtype=FP16, device=DEV)
    _keepalive.append(ones)
    bufs = {
        "context": context.data_ptr(),
        "backbone_hidden": backbone_hidden.data_ptr(),
        "txt_mlp_hidden": _zeros(x0, mlp_hidden).data_ptr(),
        "img_mlp_hidden": _zeros(a0 - x0, mlp_hidden).data_ptr(),
        "single_mlp_hidden": _zeros(a0, mlp_hidden).data_ptr(),
        "proj_scratch": _zeros(a0, hidden).data_ptr(),
        "normed_scratch": _zeros(a0, hidden).data_ptr(),
        "norm_ones": ones.data_ptr(),
    }

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=a0 + 4)
    ctx = fvk.FvkContext()
    K_cache = _zeros(num_layers, a0, HD)
    V_cache = _zeros(num_layers, a0, HD)
    Q_O = _zeros(a0, hidden)
    logits = _zeros(a0 * NH, a0 + (a0 % 2))
    layer_stride = K_cache[0].numel() * 2  # bytes
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

    assert torch.isfinite(backbone_hidden).all(), "backbone_hidden has NaN/Inf"
    assert torch.isfinite(K_cache).all(), "K cache has NaN/Inf"
    assert torch.isfinite(V_cache).all(), "V cache has NaN/Inf"
    assert (K_cache != 0).any(), "K cache was never written"
    assert (V_cache != 0).any(), "V cache was never written"
    print("PASS: imagewam_prefill finite, KV cache populated")


if __name__ == "__main__":
    test_prefill_runs_and_populates_kv_cache()

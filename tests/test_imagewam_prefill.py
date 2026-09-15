"""ImageWAM backbone prefill wiring test (plan.md Phase 3, rewritten
2026-09-14 alongside `pipeline_thor.py`'s real-math rewrite).

Random weights, small dims (not the real FLUX.2-4B size) -- this
checks `imagewam_prefill`'s own LOOPING wiring across multiple double/
single layers with DIFFERENT weights per layer: no NaN/Inf, correct
shapes, the KV cache actually gets written. Per-layer MATH correctness
(does one layer's own real math match the verified tensor-level
reference) is checked separately and more directly in
`tests/test_imagewam_thor_real_wiring.py` -- this file does not
re-litigate that, only the multi-layer loop.

`HD=128` is fixed (real 4-axis RoPE sums to 128, not a free "keep it
small" parameter -- see `_imagewam_thor_spec.py`/`imagewam_thor.py`'s
own docstrings).

Every buffer/weight tensor is kept alive in `_keepalive` for the whole
test: a bare `torch.zeros(...).data_ptr()` expression drops the only
Python reference to the tensor the instant that line finishes, and
PyTorch's caching allocator is then free to hand that same memory to
the next allocation -- silently corrupting an already-stored pointer
that nothing still references. **Found the hard way while writing this
file's own rewrite**: an earlier draft built weights as
`_rand(n, k).t().contiguous().data_ptr()` -- this hits the exact same
trap one level deeper than it looks. `.t()` returns a non-contiguous
VIEW (sharing `_rand`'s own kept-alive storage, safe on its own), but
`.contiguous()` on a non-contiguous tensor allocates a BRAND NEW
tensor (the actual GEMM-(K,N)-convention copy the weight dict needs),
which nothing kept alive before `.data_ptr()` stripped it down to a
bare int. The result was a real, silent, systematic NaN in every
double-stream layer's own MLP path -- not a `pipeline_thor.py` wiring
bug, confirmed only after exhaustively tracing every intermediate
value against an independent, correct hand-reproduction. Fixed by
`_lin()` below, which keeps the transposed `.contiguous()` result
alive via its own `_keepalive` append.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_real import compute_shared_modulation
from flash_rt.models.imagewam.pipeline_thor import imagewam_prefill
from flash_rt.models.imagewam.quant_linear import Fp16Linear
from flash_rt.models.imagewam.rope import build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_keepalive = []


def _rand(*shape, dtype=FP16, scale=0.02):
    t = (torch.randn(*shape, dtype=torch.float32, device=DEV) * scale).to(dtype)
    _keepalive.append(t)
    return t


def _lin(n, k, scale=0.02):
    """Weight in GEMM (K,N) convention: built as (n,k) then transposed
    -- `.contiguous()` on that non-contiguous transpose allocates a
    NEW tensor, which THIS helper (not the caller) keeps alive; see
    module docstring for the dangling-pointer bug this fixes."""
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


def test_prefill_runs_and_populates_kv_cache():
    torch.manual_seed(0)
    NH, HD, mlp_hidden, joint_attention_dim = 2, 128, 192, 64
    hidden = NH * HD
    num_double, num_single = 2, 3
    num_layers = num_double + num_single
    x0, a0 = 4, 8  # 4 text tokens + 4 image tokens

    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden,
                joint_attention_dim=joint_attention_dim, x0=x0, a0=a0,
                num_layers_double=num_double, num_layers_single=num_single)

    # gemm constructed before any weight so _fp16 below can wrap each
    # weight in an Fp16Linear (OPT-004 step 5 -- weights dict values
    # are now callables, not raw pointers; see pipeline_thor.py's own
    # module docstring).
    gemm = fvk.GemmRunner()

    def _fp16(n, k):
        return Fp16Linear(gemm, _lin(n, k).data_ptr(), n, k)

    weights = {}
    for L in range(num_double):
        weights[("backbone", "double", L, "txt_in.weight")] = _fp16(hidden, joint_attention_dim)
        weights[("backbone", "double", L, "img_in.weight")] = _fp16(hidden, HD)
        for prefix in ("txt", "img"):
            weights[("backbone", "double", L, f"{prefix}_qkv.weight")] = _fp16(3 * hidden, hidden)
            weights[("backbone", "double", L, f"{prefix}_proj.weight")] = _fp16(hidden, hidden)
            weights[("backbone", "double", L, f"{prefix}_mlp0.weight")] = _fp16(mlp_hidden * 2, hidden)
            weights[("backbone", "double", L, f"{prefix}_mlp2.weight")] = _fp16(hidden, mlp_hidden)
            weights[("backbone", "double", L, f"{prefix}_query_norm")] = _norm_scale(HD).data_ptr()
            weights[("backbone", "double", L, f"{prefix}_key_norm")] = _norm_scale(HD).data_ptr()
    for L in range(num_single):
        weights[("backbone", "single", L, "qkv.weight")] = _fp16(3 * hidden, hidden)
        weights[("backbone", "single", L, "mlp_in.weight")] = _fp16(mlp_hidden * 2, hidden)
        weights[("backbone", "single", L, "attn_out_proj.weight")] = _fp16(hidden, hidden)
        weights[("backbone", "single", L, "mlp_down.weight")] = _fp16(hidden, mlp_hidden)
        weights[("backbone", "single", L, "query_norm")] = _norm_scale(HD).data_ptr()
        weights[("backbone", "single", L, "key_norm")] = _norm_scale(HD).data_ptr()

    context = _rand(x0, joint_attention_dim)
    # Image rows must start with real (non-degenerate) content -- an
    # all-zero row has zero variance, which is a degenerate input for
    # LayerNorm (this project's real per-block norm, replacing the old
    # unweighted-RMS-norm-only approximation) that a real encoded
    # observation would never actually produce. Now overwritten by
    # img_in.weight's own GEMM before any LayerNorm reads it (OPT-001/
    # OPT-008) -- kept non-degenerate anyway for defense in depth.
    backbone_hidden = _rand(a0, hidden, scale=0.1)
    img_raw = _rand(a0 - x0, HD, scale=0.1)
    bufs = {
        "context": context.data_ptr(),
        "backbone_hidden": backbone_hidden.data_ptr(),
        "img_raw": img_raw.data_ptr(),
        "modded_scratch": _zeros(a0, hidden).data_ptr(),
        "txt_qkv_merged": _zeros(x0, 3 * hidden).data_ptr(),
        "img_qkv_merged": _zeros(a0 - x0, 3 * hidden).data_ptr(),
        "single_qkv_merged": _zeros(a0, 3 * hidden).data_ptr(),
        "txt_mlp_merged": _zeros(x0, mlp_hidden * 2).data_ptr(),
        "txt_mlp_gated": _zeros(x0, mlp_hidden).data_ptr(),
        "img_mlp_merged": _zeros(a0 - x0, mlp_hidden * 2).data_ptr(),
        "img_mlp_gated": _zeros(a0 - x0, mlp_hidden).data_ptr(),
        "single_mlp_merged": _zeros(a0, mlp_hidden * 2).data_ptr(),
        "single_mlp_gated": _zeros(a0, mlp_hidden).data_ptr(),
        "proj_scratch": _zeros(a0, hidden).data_ptr(),
        "proj_scratch2": _zeros(a0, hidden).data_ptr(),
    }

    mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    timestep = torch.zeros(1, device=DEV)
    mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
    rope_table = build_backbone_rope_table(x0, a0 - x0, 1, device=DEV)

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=a0 + 4,
                                         num_layers=num_layers, num_heads=NH, head_dim=HD)
    ctx = fvk.FvkContext()
    K_cache = _zeros(num_layers, a0, hidden)
    V_cache = _zeros(num_layers, a0, hidden)
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
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    imagewam_prefill(ctx, fvk, gemm, bufs, weights, dims, stream=0, attn=backend,
                      mod_txt=mod_txt, mod_img=mod_img, mod_single=mod_single,
                      rope_table=rope_table.data_ptr())
    torch.cuda.synchronize()

    assert torch.isfinite(backbone_hidden).all(), "backbone_hidden has NaN/Inf"
    assert torch.isfinite(K_cache).all(), "K cache has NaN/Inf"
    assert torch.isfinite(V_cache).all(), "V cache has NaN/Inf"
    assert (K_cache != 0).any(), "K cache was never written"
    assert (V_cache != 0).any(), "V cache was never written"
    print("PASS: imagewam_prefill finite, KV cache populated")


if __name__ == "__main__":
    test_prefill_runs_and_populates_kv_cache()

"""AWQ folds (`flash_rt/models/imagewam/awq.py`) in fp16.

Fold A: an AWQ GEMM needs input `x / s`; the AdaLN pair that produces
`x` becomes `(shift / s, (1 + scale) / s - 1)`. Measured against the
ideal `x / s` in fp32 next to the kernel's own fp16 output rounding and
next to the alternative of a separate per-channel multiply.

Fold B: a down projection's `1/s` is multiplied into the up columns of
the preceding gate/up GEMM: `silu(g) * (u / s) == (silu(g) * u) / s`.

Pipeline: at toy dims, the served pipeline with every AWQ weight
transformed (rows x s, up columns x 1/s) and the fold-A hook active, but
NO quantization (plain fp16 GEMMs), must reproduce the untransformed
pipeline up to fp16 rounding -- the end-to-end check that every fold is
wired to the right GEMM, with the gated residual + AdaLN fusion on and
off and with the merged and split single-stream linear2.
"""
import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.awq import (
    AwqScaledLinear, apply_awq_plan, awq_scale, fold_inv_scale_into_modulation, plan_awq,
)
from flash_rt.models.imagewam.quant_linear import Fp16Linear

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
FP16_UNIT_ROUNDOFF = 2.0 ** -11


def _errors(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    d = (y.float() - ref.float())
    return (d.abs().max().item(), (d.norm() / ref.float().norm()).item(),
            (d.abs() / ref.float().abs().clamp(min=1e-2)).max().item())


def test_fold_a_adaln_exactness_fp16():
    torch.manual_seed(0)
    seq, dim, eps = 905, 3072, 1e-6
    h = (torch.randn(seq, dim, device=DEV) * 50).to(BF16)       # wide residual, like backbone_hidden
    scale = torch.randn(1, 1, dim, device=DEV) * 0.5
    scale[..., :64] = -1.0 + torch.rand(64, device=DEV) * 0.02  # near-cancelling (1 + scale) ~ 0
    shift = torch.randn(1, 1, dim, device=DEV) * 0.5
    amax = torch.rand(dim, device=DEV) ** 4 * 100 + 0.01        # skewed per-channel activation amax
    s = awq_scale(amax, 0.5)
    inv_s = 1.0 / s
    print(f"s range [{s.min().item():.3f}, {s.max().item():.3f}], "
          f"clamped low/high: {(s <= 0.25).sum().item()}/{(s >= 4.0).sum().item()}")

    hf = h.float()
    ln = (hf - hf.mean(1, keepdim=True)) * torch.rsqrt(hf.var(1, unbiased=False, keepdim=True) + eps)
    ideal_x = ln * (1 + scale[0, 0]) + shift[0, 0]
    ideal_xs = ideal_x * inv_s

    def ada(sc16, sh16):
        out = torch.zeros(seq, dim, dtype=FP16, device=DEV)
        fvk.ada_layer_norm_bf16in_fp16out(h.data_ptr(), sc16.data_ptr(), sh16.data_ptr(), out.data_ptr(),
                                          seq, dim, eps, 0)
        torch.cuda.synchronize()
        return out

    plain = ada(scale[0, 0].to(FP16).contiguous(), shift[0, 0].to(FP16).contiguous())
    sh_f, sc_f = fold_inv_scale_into_modulation(shift, scale, inv_s)
    folded = ada(sc_f, sh_f)
    explicit = (plain.float() * inv_s).to(FP16)                 # separate per-channel multiply

    e_plain = _errors(plain, ideal_x)
    e_folded = _errors(folded, ideal_xs)
    e_explicit = _errors(explicit, ideal_xs)
    for name, (mx, rl, mr) in (("no AWQ (kernel fp16 rounding)", e_plain),
                               ("fold A (folded modulation)", e_folded),
                               ("separate multiply", e_explicit)):
        print(f"{name:32s} max-abs={mx:.3e} rel_l2={rl:.3e} max-rel={mr:.3e}")
    # Same order as fp16's own rounding of the unfolded output.
    assert e_folded[1] < 2 * FP16_UNIT_ROUNDOFF
    assert e_folded[1] <= 1.5 * e_plain[1]

    # The GEMM the fold exists for: x/s @ (W * s) == x @ W.
    k, n = dim, 1024
    w = (torch.randn(k, n, device=DEV) * 0.02).to(FP16)
    ws = (w.float() * s.unsqueeze(1)).to(FP16)
    y_ref = plain.float() @ w.float()
    y_awq = folded.float() @ ws.float()
    c = torch.nn.functional.cosine_similarity(y_awq.flatten(), y_ref.flatten(), dim=0).item()
    mx, rl, _ = _errors(y_awq, y_ref)
    print(f"GEMM with fold A vs without: cos={c:.8f} max-abs={mx:.3e} rel_l2={rl:.3e}")
    assert rl < 4 * FP16_UNIT_ROUNDOFF


def test_fold_b_up_columns_exactness_fp16():
    torch.manual_seed(1)
    m, k, mh = 64, 1024, 4096
    x = torch.randn(m, k, device=DEV).to(FP16)
    w_gu = (torch.randn(k, 2 * mh, device=DEV) * 0.02).to(FP16)   # [gate | up], mlp0 layout
    s_dn = awq_scale(torch.rand(mh, device=DEV) ** 4 * 50 + 0.01, 0.5)
    w_gu_b = w_gu.float()
    w_gu_b[:, mh:] *= (1.0 / s_dn).unsqueeze(0)
    w_gu_b = w_gu_b.to(FP16)
    gemm = fvk.GemmRunner()

    def gated(wt):
        merged = torch.zeros(m, 2 * mh, dtype=FP16, device=DEV)
        out = torch.zeros(m, mh, dtype=FP16, device=DEV)
        Fp16Linear(gemm, wt.data_ptr(), 2 * mh, k)(x.data_ptr(), merged.data_ptr(), m, 0)
        fvk.silu_glu_merged_fp16(merged.data_ptr(), out.data_ptr(), m, mh, 0)
        torch.cuda.synchronize()
        return out

    ideal = gated(w_gu).float() / s_dn
    got = gated(w_gu_b)
    mx, rl, mr = _errors(got, ideal)
    print(f"fold B: silu(g) * (u/s) vs (silu(g)*u)/s  max-abs={mx:.3e} rel_l2={rl:.3e}")
    assert rl < 4 * FP16_UNIT_ROUNDOFF


class _Fp16Awq(AwqScaledLinear):
    """Plain fp16 GEMM over an AWQ-transformed weight: no quantization,
    so any output difference comes from the folds themselves."""

    def __init__(self, gemm, w: torch.Tensor, n: int, k: int, inv_s: torch.Tensor | None):
        super().__init__()
        self._w = w
        self.k, self.n = k, n
        self._inner = Fp16Linear(gemm, w.data_ptr(), n, k)
        self._inv_s = inv_s

    @property
    def awq_inv_s(self):
        return self._inv_s

    def __call__(self, x_ptr, out_ptr, m, stream=0):
        self._inner(x_ptr, out_ptr, m, stream)


def _view(ptr: int, k: int, n: int) -> torch.Tensor:
    interface = {"data": (int(ptr), False), "shape": (k, n), "typestr": "<f2", "version": 3}
    return torch.as_tensor(type("_V", (), {"__cuda_array_interface__": interface})(), device=DEV)


@pytest.mark.parametrize("dims_override", [
    None,                                                  # served: fused residual+AdaLN, merged linear2
    {"fuse_res_norm": False},                              # standalone AdaLN kernels
    {"fuse_res_norm": False, "merge_linear2": False},      # split attn_out_proj / mlp_down
])
def test_pipeline_with_awq_folds_matches_unscaled_fp16(dims_override):
    torch.manual_seed(2)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dims_override)
    fe.set_prompt()
    noise = torch.randn(fe.dims["num_action"], fe.dims["action_dim"], device=DEV)
    fe.stage_inputs({}, noise=noise)
    img_raw = fe._img_raw.clone()

    fe.run_eager()
    torch.cuda.synchronize()
    ref_hidden = fe._backbone_hidden.float().clone()
    ref_latent = fe._action_latent.clone()

    linears = {k: v for k, v in fe.weights.items() if isinstance(v, Fp16Linear)}
    amax = {k: torch.rand(v.k, device=DEV) ** 4 * 20 + 0.05 for k, v in linears.items()}
    for scope in ("adaln", "adaln+down"):
        plans = plan_awq(list(linears), amax, fe.dims, alpha=0.5, scope=scope)
        weights = dict(fe.weights)
        for key, plan in plans.items():
            lin = linears[key]
            w_new = apply_awq_plan(_view(lin.weight_ptr, lin.k, lin.n), plan)
            inv_s = (1.0 / plan.input_scale) if plan.fold_input else None
            weights[key] = _Fp16Awq(fe._gemm, w_new, lin.n, lin.k, inv_s)
        n_fold_a = sum(p.fold_input for p in plans.values())
        n_fold_b = sum(p.up_inv_scale is not None for p in plans.values())
        fe.stage_inputs({}, noise=noise)
        fe._img_raw.copy_(img_raw)  # stage_inputs re-randomizes img_raw without a VAE
        fe.run_eager(weights)
        torch.cuda.synchronize()
        c_h = torch.nn.functional.cosine_similarity(fe._backbone_hidden.float().flatten(),
                                                    ref_hidden.flatten(), dim=0).item()
        c_a = torch.nn.functional.cosine_similarity(fe._action_latent.flatten(), ref_latent.flatten(),
                                                    dim=0).item()
        _, rl_a, _ = _errors(fe._action_latent, ref_latent)
        print(f"{dims_override} scope={scope}: {n_fold_a} fold-A weights, {n_fold_b} fold-B weights; "
              f"backbone_hidden cos={c_h:.7f} action_latent cos={c_a:.7f} rel_l2={rl_a:.2e}")
        assert n_fold_a > 0 and (scope == "adaln" or n_fold_b > 0)
        assert c_h > 0.9999 and c_a > 0.9999


def _count_kernels(fn) -> int:
    """CUDA kernel events CUPTI reports for one call of `fn`."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def test_fold_a_adds_no_kernels_after_first_call():
    """The folded modulation is computed once per (layer, modulation) and
    then read from the cache: after one warmup run (graph capture's own
    warmup), an AWQ pipeline launches exactly as many kernels as the
    plain one."""
    torch.manual_seed(4)
    fe = ImageWAMTorchFrontendThor(precision="fp16")
    fe.set_prompt()
    fe.stage_inputs({}, noise=torch.randn(fe.dims["num_action"], fe.dims["action_dim"], device=DEV))
    linears = {k: v for k, v in fe.weights.items() if isinstance(v, Fp16Linear)}
    amax = {k: torch.rand(v.k, device=DEV) + 0.1 for k, v in linears.items()}
    plans = plan_awq(list(linears), amax, fe.dims, alpha=0.5, scope="adaln+down")
    weights = dict(fe.weights)
    for key, plan in plans.items():
        lin = linears[key]
        weights[key] = _Fp16Awq(fe._gemm, apply_awq_plan(_view(lin.weight_ptr, lin.k, lin.n), plan),
                                lin.n, lin.k, (1.0 / plan.input_scale) if plan.fold_input else None)
    fe.run_eager()
    fe.run_eager(weights)  # warmup: fills the fold caches
    # The first profiled region in a process can report nothing at all: on Thor
    # (0919e, torch 2.9.1) the region profiling `fe.run_eager` read 0 CUDA events
    # while the region right after it read the 285 this study recorded for both
    # paths (docs/imagewam_nvfp4_awq.md). Both calls run the same `run_eager` --
    # it is eager with and without `weights` (the two differ only in the weight
    # dict) -- and `test_pipeline_with_awq_folds_matches_unscaled_fp16` shows the
    # no-weights call does the real pipeline, so that 0 is the measurement, not
    # the pipeline: a first region's count is not a kernel count. Which CUPTI
    # state makes the first region blind is not pinned down here (torch's own
    # profiler carries a CUPTI teardown / lazy-re-init workaround for CUDA
    # graphs); the test discards one region as a profiler warm-up and counts the
    # two that follow. `n_plain > 0` keeps a blind profiler from satisfying the
    # equality below on its own.
    _count_kernels(fe.run_eager)
    n_plain = _count_kernels(fe.run_eager)
    n_awq = _count_kernels(lambda: fe.run_eager(weights))
    print(f"kernels per eager forward: plain={n_plain} awq={n_awq}")
    assert n_plain > 0, "no CUDA events reported for the plain eager forward -- profiler not live"
    assert n_awq == n_plain

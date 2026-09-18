"""Roadmap item 3 (plan.md "gated-residual + next-AdaLN fusion"):
`gate_res_ada_layer_norm_{bf16res,fp16}` replaces the pair
`gate_res_{bf16res,fp16}` + `ada_layer_norm_{bf16in_fp16out,fp16}` that
ends every ImageWAM sub-block, and `dims["fuse_res_norm"]` chains it
across layer boundaries in `pipeline_thor.py`.

The fused path keeps the math unchanged, so both checks here demand
bit-exact equality with the unfused path and print the error against an
FP32 torch reference alongside:

1. Kernel level, at the real shapes: backbone BF16 residual (txt rows
   x0=513, img rows 392, single-stream rows a0=905; hidden 3072) and
   ActionDiT FP16 residual (64 x 1024).
2. Whole pass: real-dims prefill + 10-step denoise (random weights,
   fp16) with the flag off and on -- backbone residual, every layer's
   K/V cache, and the final action latent.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
F32 = torch.float32
EPS = 1e-6


def _stats(a: torch.Tensor, b: torch.Tensor) -> str:
    a_, b_ = a.float().flatten(), b.float().flatten()
    cos = (a_ @ b_ / (a_.norm() * b_.norm() + 1e-12)).item()
    rel_l2 = ((a_ - b_).norm() / (b_.norm() + 1e-12)).item()
    return f"cos={cos:.7f} max_abs={(a_ - b_).abs().max().item():.3e} rel_l2={rel_l2:.3e}"


def _modulation(dim: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(shift, scale, gate) as (1,1,dim) FP32 chunk views of one
    (1,1,3*dim) tensor -- the exact layout `adaln.modulation` returns."""
    out = torch.randn(1, 1, 3 * dim, generator=gen, device=DEV) * 0.5
    return out.chunk(3, dim=-1)


def _one_case(rows: int, dim: int, res_dtype: torch.dtype, res_scale: float, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    residual = (torch.randn(rows, dim, generator=gen, device=DEV) * res_scale).to(res_dtype)
    proj = torch.randn(rows, dim, generator=gen, device=DEV).to(FP16)
    shift, scale, gate = _modulation(dim, gen)

    # Unfused: FP16 modulation vectors (what _fuse_mod_group builds), then
    # gate_res_* over the (rows, dim) broadcast gate and ada_layer_norm_*.
    shift16, scale16 = shift[0, 0].to(FP16).contiguous(), scale[0, 0].to(FP16).contiguous()
    gate_rows = gate[0, 0].to(FP16).expand(rows, dim).contiguous()
    res_u = residual.clone()
    out_u = torch.zeros(rows, dim, dtype=FP16, device=DEV)
    # Fused: FP32 (dim,) views straight from the modulation output.
    res_f = residual.clone()
    out_f = torch.zeros(rows, dim, dtype=FP16, device=DEV)
    if res_dtype == BF16:
        fvk.gate_res_bf16res(proj.data_ptr(), gate_rows.data_ptr(), res_u.data_ptr(), rows * dim, 0)
        fvk.ada_layer_norm_bf16in_fp16out(res_u.data_ptr(), scale16.data_ptr(), shift16.data_ptr(),
                                          out_u.data_ptr(), rows, dim, EPS, 0)
        fvk.gate_res_ada_layer_norm_bf16res(proj.data_ptr(), gate.data_ptr(), res_f.data_ptr(),
                                            scale.data_ptr(), shift.data_ptr(), out_f.data_ptr(),
                                            rows, dim, EPS, 0)
    else:
        fvk.gate_res_fp16(proj.data_ptr(), gate_rows.data_ptr(), res_u.data_ptr(), rows * dim, 0)
        fvk.ada_layer_norm_fp16(res_u.data_ptr(), scale16.data_ptr(), shift16.data_ptr(),
                                out_u.data_ptr(), rows, dim, EPS, 0)
        fvk.gate_res_ada_layer_norm_fp16(proj.data_ptr(), gate.data_ptr(), res_f.data_ptr(),
                                         scale.data_ptr(), shift.data_ptr(), out_f.data_ptr(),
                                         rows, dim, EPS, 0)
    torch.cuda.synchronize()

    # FP32 torch reference (FP16 modulation values, residual stored in its dtype).
    ref_res = (residual.float() + proj.float() * gate_rows.float()).to(res_dtype)
    r = ref_res.float()
    mean = r.mean(dim=-1, keepdim=True)
    var = ((r - mean) ** 2).mean(dim=-1, keepdim=True)
    ref_out = (r - mean) * torch.rsqrt(var + EPS) * (1 + scale16.float()) + shift16.float()
    return res_f, out_f, res_u, out_u, ref_res, ref_out


def test_fused_kernel_bit_exact_vs_unfused_real_shapes():
    cases = {
        "backbone txt rows (bf16 res)": (513, 3072, BF16, 3.0e4),
        "backbone img rows (bf16 res)": (392, 3072, BF16, 4.0),
        "backbone single rows (bf16 res)": (905, 3072, BF16, 4.0),
        "actiondit rows (fp16 res)": (64, 1024, FP16, 4.0),
    }
    for i, (name, (rows, dim, res_dtype, res_scale)) in enumerate(cases.items()):
        res_f, out_f, res_u, out_u, ref_res, ref_out = _one_case(rows, dim, res_dtype, res_scale, seed=i)
        res_exact, out_exact = torch.equal(res_f, res_u), torch.equal(out_f, out_u)
        print(f"{name} {rows}x{dim}: fused vs unfused residual bit_exact={res_exact} "
              f"normed bit_exact={out_exact}")
        print(f"    fused vs fp32 torch: residual {_stats(res_f, ref_res)}; normed {_stats(out_f, ref_out)}")
        assert res_exact and out_exact
        assert bool(torch.isfinite(out_f.float()).all())


def test_fused_kernel_residual_only_mode():
    """`out == nullptr` (no AdaLN follows -- the backbone's last layer):
    residual update only, bit-exact vs `gate_res_bf16res`."""
    rows, dim = 905, 3072
    gen = torch.Generator(device=DEV).manual_seed(7)
    residual = (torch.randn(rows, dim, generator=gen, device=DEV) * 4.0).to(BF16)
    proj = torch.randn(rows, dim, generator=gen, device=DEV).to(FP16)
    _, _, gate = _modulation(dim, gen)
    gate_rows = gate[0, 0].to(FP16).expand(rows, dim).contiguous()
    res_u, res_f = residual.clone(), residual.clone()
    fvk.gate_res_bf16res(proj.data_ptr(), gate_rows.data_ptr(), res_u.data_ptr(), rows * dim, 0)
    fvk.gate_res_ada_layer_norm_bf16res(proj.data_ptr(), gate.data_ptr(), res_f.data_ptr(), 0, 0, 0,
                                        rows, dim, EPS, 0)
    torch.cuda.synchronize()
    exact = torch.equal(res_f, res_u)
    print(f"residual-only mode {rows}x{dim}: fused vs gate_res_bf16res bit_exact={exact}")
    assert exact


def test_fused_kernel_rejects_odd_dim():
    """The kernel reads rows as packed pairs; the launcher rejects an odd
    `dim` (and an empty row count) on the host instead of faulting."""
    import pytest
    buf = torch.zeros(4, 8, dtype=FP16, device=DEV)
    vec = torch.zeros(8, dtype=F32, device=DEV)
    for fn in (fvk.gate_res_ada_layer_norm_bf16res, fvk.gate_res_ada_layer_norm_fp16):
        for rows, dim in ((4, 7), (0, 8)):
            with pytest.raises(ValueError, match="even dim"):
                fn(buf.data_ptr(), vec.data_ptr(), buf.data_ptr(), vec.data_ptr(), vec.data_ptr(),
                   buf.data_ptr(), rows, dim, EPS, 0)
    torch.cuda.synchronize()
    print("gate_res_ada_layer_norm: odd dim / zero rows -> ValueError, no launch")


REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / 10, num_denoise_steps=10,
    ref_h=14, ref_w=28, shift=5.0, num_train_timesteps=1000,
)


def _pipeline_pass(fe: ImageWAMTorchFrontendThor, fuse: bool, ctx: torch.Tensor, img: torch.Tensor,
                   noise: torch.Tensor) -> dict:
    """One eager prefill + denoise pass with `dims["fuse_res_norm"]=fuse`
    over the frontend's own weights/buffers, from identical inputs."""
    fe.dims["fuse_res_norm"] = fuse
    fe._context.copy_(ctx)
    fe._img_raw.copy_(img)
    fe._action_latent.copy_(noise)
    s = torch.cuda.current_stream().cuda_stream
    imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
                     mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                     rope_table=fe._rope_table.data_ptr())
    backbone_hidden = fe._backbone_hidden.clone()
    imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
                          action_mods=fe._action_mods, head_mods=fe._head_mods,
                          action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)
    torch.cuda.synchronize()
    return {"backbone_hidden": backbone_hidden, "K_cache": fe._K_cache.clone(),
            "V_cache": fe._V_cache.clone(), "action_latent": fe._action_latent.clone()}


def test_pipeline_fused_bit_exact_vs_unfused_real_dims():
    """Real-dims prefill (5 double + 20 single) + 10-step denoise (5 + 20
    ActionDiT layers, head), random weights, fp16 (merged linear1/linear2
    on): the fused chain -- including the double->single and
    single->head boundaries -- reproduces the unfused pass bit for bit."""
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(REAL_DIMS))
    gen = torch.Generator(device=DEV).manual_seed(3)
    x0, a0 = REAL_DIMS["x0"], REAL_DIMS["a0"]
    ctx = (torch.randn(x0, REAL_DIMS["joint_attention_dim"], generator=gen, device=DEV)).to(BF16)
    img = torch.randn(a0 - x0, REAL_DIMS["HD"], generator=gen, device=DEV).to(BF16)
    noise = torch.randn(REAL_DIMS["num_action"], fe.dims["action_dim"], generator=gen, device=DEV)

    ref = _pipeline_pass(fe, False, ctx, img, noise)
    got = _pipeline_pass(fe, True, ctx, img, noise)
    for name in ("backbone_hidden", "K_cache", "V_cache", "action_latent"):
        exact = torch.equal(got[name], ref[name])
        finite = bool(torch.isfinite(got[name].float()).all())
        print(f"real-dims pass, fused vs unfused {name} {tuple(got[name].shape)}: bit_exact={exact} "
              f"finite={finite} {_stats(got[name], ref[name])}")
        assert exact and finite

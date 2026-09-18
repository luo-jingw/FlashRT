"""`precision="e0m3_hadamard"`: routing (runs anywhere) and the Thor-only
kernel path of `E0m3HadamardLinear` against `blockscaled_ref.py`.

- Routing: every 16-aligned weight goes to `E0m3HadamardLinear`, the
  `K=7` `action_encoder` and `N=7` `head.linear` fall back to
  `Fp16Linear`, `txt_in`/`img_in` stay `Bf16OutLinear`, the single-stream
  blocks keep the merged `linear1`, and the default precision stays
  `nvfp4`. The class is replaced by a recording stub, so no NVFP4 build
  is needed.
- Thor (skipped without `flash_rt.flash_rt_fp4`): the packed weight and
  its SFB, and the rotated activation quantizer's packed/SFA bytes, are
  compared byte for byte with the PyTorch reference; the GEMM is
  compared with the reference product of the dequantized operands and
  with the fp32 product of the unquantized operands, at ImageWAM's real
  shapes and for every tile variant.
"""
import inspect

import numpy as np
import pytest
import torch

import flash_rt.frontends.torch.imagewam_thor as imagewam_thor
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.blockscaled_ref import (
    dequantize_blocks,
    fwht16_butterfly,
    pack_codes,
    pack_scales,
    quantize_blocks,
)
from flash_rt.models.imagewam.quant_linear import Bf16OutLinear, Fp16Linear

DEV = "cuda"
FP16 = torch.float16

try:
    import flash_rt.flash_rt_fp4 as fvk_fp4
    _HAS_FP4 = hasattr(fvk_fp4, "cutlass_fp4_gemm_e0m3w_variant")
except ImportError:
    fvk_fp4 = None
    _HAS_FP4 = False
thor_only = pytest.mark.skipif(not _HAS_FP4, reason="needs the SM110 flash_rt_fp4 build (Thor)")

# (M, N, K) of every quantized ImageWAM GEMM family at real dims.
REAL_SHAPES = [
    (905, 27648, 3072),   # backbone single linear1 (merged qkv + mlp gate/up)
    (905, 3072, 3072),    # backbone single attn_out_proj
    (905, 3072, 9216),    # backbone single mlp_down
    (513, 9216, 3072),    # backbone double txt_qkv
    (392, 18432, 3072),   # backbone double img_mlp0
    (392, 3072, 9216),    # backbone double img_mlp2
    (64, 17408, 1024),    # ActionDiT single linear1
    (64, 1024, 3072),     # ActionDiT attn_out_proj / proj
    (64, 1024, 4096),     # ActionDiT mlp_down / mlp2
    (64, 9216, 1024),     # ActionDiT double qkv
]


class _StubLinear:
    def __init__(self, weight_fp16_ptr: int, n: int, k: int):
        self.n, self.k = n, k


def test_precision_routing(monkeypatch):
    monkeypatch.setattr(imagewam_thor, "E0m3HadamardLinear", _StubLinear)
    assert "e0m3_hadamard" in imagewam_thor._PRECISIONS
    default = inspect.signature(ImageWAMTorchFrontendThor.__init__).parameters["precision"].default
    assert default == "nvfp4"
    fe = ImageWAMTorchFrontendThor(precision="e0m3_hadamard")
    assert fe.dims["merge_qkv_mlp"] is True
    counts = {"e0m3": 0, "fp16": 0, "bf16out": 0}
    for key, v in fe._weights.items():
        slot = key[-1]
        if isinstance(v, int):
            continue
        if slot in ("action_encoder.weight", "head.linear.weight"):
            assert isinstance(v, Fp16Linear), key
            counts["fp16"] += 1
        elif slot in ("txt_in.weight", "img_in.weight"):
            assert isinstance(v, Bf16OutLinear), key
            counts["bf16out"] += 1
        else:
            assert isinstance(v, _StubLinear), key
            assert v.n % 16 == 0 and v.k % 16 == 0
            counts["e0m3"] += 1
    linear1 = [k for k in fe._weights if k[-1] == "linear1.weight"]
    assert linear1 and all(isinstance(fe._weights[k], _StubLinear) for k in linear1)
    print(f"routing: {counts}, merged linear1 weights: {len(linear1)}")


def _outlier_weight(n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.025
    w[:, torch.randint(0, k, (k // 128,), generator=g)] *= 8.0
    return w.to(FP16).t().contiguous().to(DEV)  # (K, N), the project's GEMM storage convention


def _outlier_act(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    x[:, torch.randint(0, k, (k // 256,), generator=g)] *= 30.0
    return x.to(FP16).to(DEV)


def _ref_weight(w_kn: torch.Tensor, alpha: float):
    """Reference bytes for the weight: fp32 rotation, fp16 storage scaled
    by 1/alpha, E0M3 quantization."""
    w_in = (fwht16_butterfly(w_kn.t().float()) / alpha).half()
    return quantize_blocks(w_in, "e0m3")


def _stats(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, str]:
    y, ref = y.double().flatten(), ref.double().flatten()
    c = float(y @ ref / (y.norm() * ref.norm()))
    return c, (f"cos={c:.7f} max_abs={float((y - ref).abs().max()):.3e} "
               f"rel_l2={float((y - ref).norm() / ref.norm()):.3e}")


@thor_only
@pytest.mark.parametrize("n,k", [(3072, 9216), (27648, 3072), (17408, 1024), (1024, 4096)])
def test_weight_packing_bit_exact(n, k):
    from flash_rt.models.imagewam.quant_linear import E0m3HadamardLinear
    w = _outlier_weight(n, k, 0)
    lin = E0m3HadamardLinear(w.data_ptr(), n, k)
    q = _ref_weight(w, lin.alpha)
    bad_codes = int((lin.w_packed != pack_codes(q.codes)).sum())
    bad_sf = int((lin.w_sfb != pack_scales(q.scale_bytes)).sum())
    print(f"N={n} K={k}: alpha={lin.alpha:g} packed mismatches={bad_codes}/{lin.w_packed.numel()} "
          f"sfb mismatches={bad_sf}/{lin.w_sfb.numel()}")
    assert bad_codes == 0 and bad_sf == 0


@thor_only
@pytest.mark.parametrize("m,k", [(905, 3072), (905, 9216), (64, 1024), (64, 4096), (513, 3072)])
def test_rotated_activation_quantizer_bit_exact(m, k):
    x = _outlier_act(m, k, 1)
    packed = torch.empty(m, k // 2, dtype=torch.uint8, device=DEV)
    sfa = torch.zeros(fvk_fp4.sfa_size_bytes(m, k, False), dtype=torch.uint8, device=DEV)
    rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16_vec(x.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
                                                   m, k, False, 1, 0)
    torch.cuda.synchronize()
    assert rc == 0
    q = quantize_blocks(fwht16_butterfly(x), "e0m3")
    bad_codes = int((packed != pack_codes(q.codes)).sum())
    bad_sf = int((sfa != pack_scales(q.scale_bytes)).sum())
    print(f"M={m} K={k}: packed mismatches={bad_codes}/{packed.numel()} sfa mismatches={bad_sf}/{sfa.numel()}")
    assert bad_codes == 0 and bad_sf == 0


@thor_only
@pytest.mark.parametrize("m,n,k", REAL_SHAPES)
def test_gemm_matches_reference(m, n, k):
    from flash_rt.models.imagewam.quant_linear import E0m3HadamardLinear
    w = _outlier_weight(n, k, 2)
    x = _outlier_act(m, k, 3)
    lin = E0m3HadamardLinear(w.data_ptr(), n, k)
    out = torch.empty(m, n, dtype=FP16, device=DEV)
    lin(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()
    # Reference product of the dequantized operands (what the tensor core computes).
    wq = dequantize_blocks(_ref_weight(w, lin.alpha)) * lin.alpha
    xq = dequantize_blocks(quantize_blocks(fwht16_butterfly(x), "e0m3"))
    ref_q = xq @ wq.t()
    ref_exact = x.float() @ w.float()
    cos_q, s_q = _stats(out, ref_q)
    _, s_exact = _stats(out, ref_exact)
    print(f"M={m} N={n} K={k} variant={lin.variant}: vs dequantized-operand reference {s_q}; "
          f"vs exact fp32 product {s_exact}")
    assert torch.isfinite(out).all()
    assert cos_q > 0.9999
    # Every tile variant on the same packed operands: rc and max |diff| vs
    # the dispatched variant (accumulation order differs across K tiles).
    report = {}
    for v in (1, 6, 8, 10):
        o = torch.empty_like(out)
        rc = fvk_fp4.cutlass_fp4_gemm_e0m3w_variant(
            v, lin.a_packed.data_ptr(), lin.a_sfa.data_ptr(), lin.w_packed.data_ptr(), lin.w_sfb.data_ptr(),
            o.data_ptr(), m, n, k, lin.alpha, 0.0, 0, 0)
        torch.cuda.synchronize()
        report[v] = (rc, float((o.float() - out.float()).abs().max()) if rc == 0 else None)
    print(f"   variants (rc, max |diff| vs dispatched): {report}")
    assert report[lin.variant][0] == 0 and report[lin.variant][1] == 0.0


@thor_only
def test_frontend_small_dims_end_to_end():
    fe = ImageWAMTorchFrontendThor(precision="e0m3_hadamard")
    fe.set_prompt("pick up the red cup")
    for _ in range(3):
        actions = fe.infer({"image": np.zeros((4, 4, 3), dtype=np.uint8)})["actions"]
        assert actions.shape == (fe.dims["num_action"], fe.dims["action_dim"])
        assert np.isfinite(actions).all()

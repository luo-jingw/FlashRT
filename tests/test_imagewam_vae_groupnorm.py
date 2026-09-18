"""`imagewam_groupnorm_nhwc_bf16` (roadmap item 5 phase 4, `plan.md`)
against torch `F.group_norm` (+ the FLUX.2 `x * torch.sigmoid(x)` swish)
at every GroupNorm shape of the real FLUX.2 AutoEncoder encoder for the
224x448 input.

Observational: prints cosine, max-abs, rel_l2 and the fraction of BF16
elements that differ from torch. The kernel keeps the math of torch's
three ops (Welford statistics, a = rstd*gamma, b = beta - mean*a,
y = x*a + b, BF16 rounding after norm, sigmoid and product) but reduces
in a different order and uses `__expf`, so it is near-exact rather than
bit-exact; the asserts are loose sanity bounds, not performance or
accuracy targets.

A second test feeds the real encoder's own GroupNorm inputs (captured
with forward hooks on a real LIBERO frame) when the real AE is present.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
BF16 = torch.bfloat16
G = 32
EPS = 1e-6

# (C, H, W, silu) for every GroupNorm call of AutoEncoder.encode at 224x448.
REAL_SHAPES = [
    (128, 224, 448, True), (128, 112, 224, True), (256, 112, 224, True),
    (256, 56, 112, True), (512, 56, 112, True), (512, 28, 56, True), (512, 28, 56, False),
]


def groupnorm_nhwc(x_nhwc: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, silu: bool,
                   bias: torch.Tensor | None = None) -> torch.Tensor:
    n, h, w, c = x_nhwc.shape
    y = torch.empty_like(x_nhwc)
    nbytes = fvk.imagewam_groupnorm_nhwc_workspace_bytes(n, h * w, c, G)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    rc = fvk.imagewam_groupnorm_nhwc_bf16(x_nhwc.data_ptr(), 0 if bias is None else bias.data_ptr(),
                                           gamma.data_ptr(), beta.data_ptr(), y.data_ptr(),
                                           ws.data_ptr(), nbytes, n, h * w, c, G, EPS, int(silu),
                                           torch.cuda.current_stream().cuda_stream)
    assert rc == 0, rc
    return y


def reference(x_nchw: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, silu: bool) -> torch.Tensor:
    y = F.group_norm(x_nchw, G, gamma, beta, EPS)
    if silu:
        y = y * torch.sigmoid(y)
    return y


def _report(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a, b = out.float().flatten(), ref.float().flatten()
    cos = (a @ b / (a.norm() * b.norm() + 1e-12)).item()
    maxabs = (a - b).abs().max().item()
    rel_l2 = ((a - b).norm() / (b.norm() + 1e-12)).item()
    frac = (out.view(torch.int16) != ref.view(torch.int16)).float().mean().item()
    print(f"{name}: cosine={cos:.8f} max_abs={maxabs:.3e} rel_l2={rel_l2:.3e} differing_bf16={frac * 100:.3f}%")
    return cos, rel_l2


@pytest.mark.parametrize("c,h,w,silu", REAL_SHAPES)
def test_groupnorm_nhwc_matches_torch_synthetic(c: int, h: int, w: int, silu: bool):
    g = torch.Generator(device=DEV).manual_seed(c * 7 + h)
    # Per-channel offsets and scales so the group mean is not ~0.
    offset = torch.randn(1, c, 1, 1, device=DEV, generator=g) * 2.0
    scale = torch.rand(1, c, 1, 1, device=DEV, generator=g) * 3.0 + 0.1
    x = (torch.randn(1, c, h, w, device=DEV, generator=g) * scale + offset).to(BF16)
    gamma = (torch.randn(c, device=DEV, generator=g) * 0.5 + 1.0).to(BF16)
    beta = (torch.randn(c, device=DEV, generator=g) * 0.2).to(BF16)
    ref = reference(x, gamma, beta, silu)
    out = groupnorm_nhwc(x.permute(0, 2, 3, 1).contiguous(), gamma, beta, silu)
    cos, rel = _report(f"GN{'+SiLU' if silu else ''} C={c} {h}x{w}", out, ref.permute(0, 2, 3, 1))
    assert cos > 0.9999 and rel < 1e-2


def test_groupnorm_nhwc_in_place():
    x = torch.randn(1, 28, 56, 512, device=DEV).to(BF16)
    gamma = torch.ones(512, device=DEV, dtype=BF16)
    beta = torch.zeros(512, device=DEV, dtype=BF16)
    ref = groupnorm_nhwc(x, gamma, beta, True)
    y = x.clone()
    nbytes = fvk.imagewam_groupnorm_nhwc_workspace_bytes(1, 28 * 56, 512, G)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    assert fvk.imagewam_groupnorm_nhwc_bf16(y.data_ptr(), 0, gamma.data_ptr(), beta.data_ptr(), y.data_ptr(),
                                             ws.data_ptr(), nbytes, 1, 28 * 56, 512, G, EPS, 1, 0) == 0
    torch.cuda.synchronize()
    assert torch.equal(y, ref)


def test_groupnorm_nhwc_rejects_unsupported():
    assert fvk.imagewam_groupnorm_nhwc_workspace_bytes(1, 64, 96, 32) == 0  # C/G = 3
    assert fvk.imagewam_groupnorm_nhwc_bf16(0, 0, 0, 0, 0, 0, 0, 1, 64, 96, 32, EPS, 0, 0) < 0


@pytest.mark.parametrize("c,h,w", [(256, 112, 224), (512, 28, 56)])
def test_groupnorm_nhwc_fused_input_bias(c: int, h: int, w: int):
    """GroupNorm(bf16(x + bias)) vs torch's separate bias add then GroupNorm+swish."""
    g = torch.Generator(device=DEV).manual_seed(c)
    x = torch.randn(1, c, h, w, device=DEV, generator=g).to(BF16)
    bias = (torch.randn(c, device=DEV, generator=g) * 0.5).to(BF16)
    gamma = (torch.randn(c, device=DEV, generator=g) * 0.5 + 1.0).to(BF16)
    beta = (torch.randn(c, device=DEV, generator=g) * 0.2).to(BF16)
    ref = reference(x + bias.view(1, c, 1, 1), gamma, beta, True)
    out = groupnorm_nhwc(x.permute(0, 2, 3, 1).contiguous(), gamma, beta, True, bias=bias)
    cos, rel = _report(f"GN+SiLU(x+bias) C={c} {h}x{w}", out, ref.permute(0, 2, 3, 1))
    assert cos > 0.9999 and rel < 1e-2


@pytest.mark.parametrize("shortcut_bias", [False, True])
def test_bias_residual_nhwc_bitexact(shortcut_bias: bool):
    """(res [+ res_bias]) + (h + h_bias) vs torch's separate BF16 adds: bit-exact."""
    g = torch.Generator(device=DEV).manual_seed(3)
    c, h, w = 256, 112, 224
    hv = torch.randn(1, h, w, c, device=DEV, generator=g).to(BF16)
    res = torch.randn(1, h, w, c, device=DEV, generator=g).to(BF16)
    hb = torch.randn(c, device=DEV, generator=g).to(BF16)
    rb = torch.randn(c, device=DEV, generator=g).to(BF16)
    ref = ((res + rb) if shortcut_bias else res) + (hv + hb)
    y = torch.empty_like(hv)
    rc = fvk.imagewam_bias_residual_nhwc_bf16(hv.data_ptr(), hb.data_ptr(), res.data_ptr(),
                                              rb.data_ptr() if shortcut_bias else 0, y.data_ptr(), h * w, c, 0)
    assert rc == 0
    torch.cuda.synchronize()
    _report(f"bias+residual shortcut_bias={shortcut_bias}", y, ref)
    assert torch.equal(y, ref)


_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH", "")


@pytest.mark.skipif(not (os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)), reason="real AE not present")
def test_groupnorm_nhwc_on_real_encoder_activations():
    """Every GroupNorm of the real encoder, fed its real input from a
    real (or synthetic, without DATA_ROOT) 224x448 image."""
    from flash_rt.models.imagewam.vae_encoder import load_real_ae
    from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor
    from test_imagewam_vae_stage import _frames

    ae = load_real_ae(_AE_PATH, _FLUX2_SRC)
    frames = _frames(4)
    views = [torch.from_numpy(frames[0]).to(DEV), torch.from_numpy(frames[2]).to(DEV)]
    image = torch.empty(1, 3, 224, 448, dtype=BF16, device=DEV)
    VaePreprocessor().run(views, image, 0)
    captured: list[tuple[str, torch.nn.GroupNorm, torch.Tensor]] = []
    hooks = [m.register_forward_hook(lambda mod, inp, out, name=name: captured.append((name, mod, inp[0].clone())))
             for name, m in ae.encoder.named_modules() if isinstance(m, torch.nn.GroupNorm)]
    with torch.no_grad():
        ae.encode(image)
    for h in hooks:
        h.remove()
    assert len(captured) == 22
    worst = 1.0
    for name, mod, x in captured:
        silu = not name.endswith("attn_1.norm")
        ref = reference(x, mod.weight, mod.bias, silu)
        out = groupnorm_nhwc(x.permute(0, 2, 3, 1).contiguous(), mod.weight, mod.bias, silu)
        cos, _ = _report(f"real {name} {tuple(x.shape)}", out, ref.permute(0, 2, 3, 1))
        worst = min(worst, cos)
    assert worst > 0.9999

"""FA4 dispatch in `ImageWAMAttnBackend` and the frontend's FA4 default
(plan.md, "Plan: attention-chain fusion recheck at ImageWAM's real
shapes", roadmap item 6; opportunities.md OPT-019).

FA4 itself only runs on Thor (`tests/test_imagewam_fa4_backbone.py`
covers the real kernel there). What can go wrong locally is the
dispatch around it: tensor views over the raw Q/K/V pointers, the Q row
offset of the "mot" site, the output staging through `logits`, and the
copy back. These tests replace `fa4_backend.fa4_fwd()` with a stand-in
that has `_flash_attn_fwd`'s calling convention and computes attention
in fp32 PyTorch, then compare each FA4 branch against the cuBLAS chain
(`attention_qkv_fp16_perhead`) at the real shapes, and run a
small-dims frontend end to end with both sites on the stand-in.
"""
from __future__ import annotations

import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor import fa4_backend
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

FP16 = torch.float16
NH, HD = 24, 128


def _fa4_stand_in(calls: list[dict]):
    def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, num_splits: int,
            pack_gqa: bool, out: torch.Tensor, softmax_scale: float | None = None):
        calls.append(dict(q=tuple(q.shape), k=tuple(k.shape), causal=causal, pack_gqa=pack_gqa,
                          num_splits=num_splits, scale=softmax_scale,
                          stream=torch.cuda.current_stream().cuda_stream))
        scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
        qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))  # (B, H, S, D)
        o = torch.softmax(qf @ kf.transpose(-1, -2) * scale, dim=-1) @ vf
        out.copy_(o.transpose(1, 2).to(out.dtype))
        return out, None
    return fwd


@pytest.fixture
def fa4_calls(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(fa4_backend, "fa4_fwd", lambda: _fa4_stand_in(calls))
    return calls


def _stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af, bf = a.float().flatten(), b.float().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-12))
    return cos, float((af - bf).abs().max()), float((af - bf).norm() / (bf.norm() + 1e-12))


def _backend(total: int, a0: int, *, use_fa4: bool, use_fa4_mot: bool, layers: int = 2, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = NH * HD
    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total,
                                        num_layers=layers, num_heads=NH, head_dim=HD)
    q_o = torch.randn(total, hidden, generator=gen, device="cuda").to(FP16)
    k = torch.randn(layers, total, hidden, generator=gen, device="cuda").to(FP16)
    v = torch.randn(layers, total, hidden, generator=gen, device="cuda").to(FP16)
    logits = torch.zeros(total * NH, total + total % 2, dtype=FP16, device="cuda")
    slots = {"Q_O": q_o.data_ptr(), "K": k.data_ptr(), "V": v.data_ptr(),
             "logits": logits.data_ptr(), "scale": HD ** -0.5}
    backend = ImageWAMAttnBackend(
        spec, fvk.FvkContext(), backbone_slots=dict(slots),
        mot_slots=dict(slots, layer_stride=k[0].numel() * 2),
        use_fa4=use_fa4, use_perhead_kv=True, use_real_mot_mask=True, use_fa4_mot=use_fa4_mot)
    return backend, (q_o, k, v, logits)


def test_backbone_fa4_branch_matches_cublas_chain(fa4_calls):
    a0, total = 905, 969
    ref_be, ref_bufs = _backend(total, a0, use_fa4=False, use_fa4_mot=False)
    fa4_be, fa4_bufs = _backend(total, a0, use_fa4=True, use_fa4_mot=False)
    stream = torch.cuda.Stream()
    ref_be.run("backbone", 1, q_seq=a0, kv_seq=a0, stream=0)
    with torch.cuda.stream(stream):
        fa4_be.run("backbone", 1, q_seq=a0, kv_seq=a0, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    cos, max_abs, rel = _stats(fa4_bufs[0][:a0], ref_bufs[0][:a0])
    print(f"\nbackbone q=kv={a0}: cosine={cos:.6f} max_abs={max_abs:.3e} rel_l2={rel:.3e}")
    assert fa4_calls[-1]["q"] == (1, a0, NH, HD) and fa4_calls[-1]["k"] == (1, a0, NH, HD)
    assert fa4_calls[-1]["pack_gqa"] is False and fa4_calls[-1]["causal"] is False
    assert fa4_calls[-1]["stream"] == stream.cuda_stream
    assert cos > 0.9999


def test_mot_fa4_branch_matches_cublas_chain(fa4_calls):
    a0, total, num_action = 905, 969, 64
    ref_be, ref_bufs = _backend(total, a0, use_fa4=False, use_fa4_mot=False)
    fa4_be, fa4_bufs = _backend(total, a0, use_fa4=False, use_fa4_mot=True)
    before = fa4_bufs[0].clone()
    ref_be.run("mot", 1, q_seq=num_action, kv_seq=total, stream=0, x0=513, a0=a0)
    fa4_be.run("mot", 1, q_seq=num_action, kv_seq=total, stream=0, x0=513, a0=a0)
    torch.cuda.synchronize()
    rows = slice(a0, a0 + num_action)
    cos, max_abs, rel = _stats(fa4_bufs[0][rows], ref_bufs[0][rows])
    print(f"\nmot q={num_action} at row {a0}, kv={total}: cosine={cos:.6f} max_abs={max_abs:.3e} rel_l2={rel:.3e}")
    assert fa4_calls[-1]["q"] == (1, num_action, NH, HD) and fa4_calls[-1]["k"] == (1, total, NH, HD)
    assert fa4_calls[-1]["scale"] == pytest.approx(HD ** -0.5)
    assert cos > 0.9999
    assert torch.equal(fa4_bufs[0][:a0], before[:a0])  # prefix/image rows untouched


def test_use_fa4_mot_requires_the_unmasked_perhead_rule(fa4_calls):
    spec = make_imagewam_attention_spec(max_prefix_seq=8, max_total_seq=12, num_layers=1,
                                        num_heads=NH, head_dim=HD)
    buf = torch.zeros(16, NH * HD, dtype=FP16, device="cuda")
    slots = {"Q_O": buf.data_ptr(), "K": buf.data_ptr(), "V": buf.data_ptr(),
             "logits": buf.data_ptr(), "scale": 1.0}
    for perhead, real_mask in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="use_fa4_mot"):
            ImageWAMAttnBackend(spec, fvk.FvkContext(), backbone_slots=dict(slots),
                                mot_slots=dict(slots, layer_stride=0), use_perhead_kv=perhead,
                                use_real_mot_mask=real_mask, use_fa4_mot=True)


def test_thor_default_resolution_on_this_device():
    major = torch.cuda.get_device_capability()[0]
    enabled = fa4_backend.thor_default_enabled()
    print(f"\ncapability major={major}, fa4 status={fa4_backend.status() if major == 11 else 'not probed'}, "
          f"thor_default_enabled={enabled}")
    if major != 11:
        assert enabled is False


def _frontend(**kw):
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    torch.manual_seed(0)
    return ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(num_action=16, total=24), **kw)


@pytest.mark.parametrize("env,runtime,arg,expected", [
    (None, True, None, False),    # default: opt-in, off even where FA4 would work
    ("0", True, None, False),
    ("1", True, None, True),      # FLASHRT_THOR_FA4=1 opts in where FA4 works
    ("1", False, None, False),    # ... and never raises where it does not
    (None, True, True, True),     # explicit argument wins over the environment
    ("1", True, False, False),
])
def test_frontend_fa4_resolution(monkeypatch, fa4_calls, env, runtime, arg, expected):
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    if env is None:
        monkeypatch.delenv("FLASHRT_THOR_FA4", raising=False)
    else:
        monkeypatch.setenv("FLASHRT_THOR_FA4", env)
    monkeypatch.setattr(fa4_backend, "thor_default_enabled", lambda: runtime)
    assert ImageWAMTorchFrontendThor._resolve_use_fa4(arg) is expected


def test_frontend_default_is_the_cublas_chain(monkeypatch, fa4_calls):
    monkeypatch.delenv("FLASHRT_THOR_FA4", raising=False)
    monkeypatch.setattr(fa4_backend, "thor_default_enabled", lambda: True)
    fe = _frontend()
    assert fe.use_fa4 is False and fe._attn._use_fa4 is False
    assert fe.use_fa4_mot is False and fe._attn._use_fa4_mot is False
    monkeypatch.setenv("FLASHRT_THOR_FA4", "1")
    fe = _frontend()
    assert fe.use_fa4 is True and fe._attn._use_fa4 is True


def test_frontend_end_to_end_with_fa4_on_both_sites(fa4_calls):
    """Same seed, same random weights: the captured graph with both sites on
    the FA4 stand-in against the cuBLAS-chain graph."""
    outs = {}
    for name, kw in (("chain", dict(use_fa4=False)), ("fa4", dict(use_fa4=True, use_fa4_mot=True))):
        fe = _frontend(**kw)
        torch.manual_seed(1)
        fe.set_prompt("fa4 dispatch")
        torch.manual_seed(2)
        outs[name] = torch.from_numpy(fe.infer({})["actions"]).float()
        del fe
    sites = {c["q"][1] for c in fa4_calls}
    cos, max_abs, rel = _stats(outs["fa4"], outs["chain"])
    print(f"\nactions fa4 vs chain: cosine={cos:.6f} max_abs={max_abs:.3e} rel_l2={rel:.3e}; "
          f"FA4 q lengths seen={sorted(sites)}")
    assert sites == {8, 16}  # backbone a0=8, mot num_action=16 (default small dims)
    assert torch.isfinite(outs["fa4"]).all()
    assert cos > 0.999

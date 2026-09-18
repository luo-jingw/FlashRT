"""Text-context trimming (`text_trim=True`, issues.md ISSUE-020) and the
`set_prompt(context=...)` cache fix (issues.md ISSUE-060).

What is checked, and against what:

- `pack_trimmed_context` against official `ImageWAM._append_proprio_to_context`
  (called directly when the `imagewam` package imports) and
  `trimmed_sequence_dims` at the real dims.
- The served per-head attention (`ImageWAMAttnBackend`, both sites) at
  trimmed real shapes, odd and even lengths, against an fp32 PyTorch
  reference, and the FA4 dispatch at those shapes with an fp32 stand-in
  for the FA4 kernel; on Thor, with the FA4 kernel itself (`fa4_real`,
  `real`, skipped without an FA4 runtime).
- A small-dims frontend with random weights: the trimmed sequence against
  the untrimmed sequence with the padded text keys masked (the official
  rule), both with an fp32 PyTorch attention, so the only difference is
  the dropped rows; prompt switching between lengths and back; lazily
  grown GEMM scratch; ISSUE-060; the VAE stage inside every length's
  graph (real AE, skipped without it).

Every numeric comparison prints cosine, max-abs and rel_l2.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.hardware.thor import fa4_backend
from flash_rt.hardware.thor.attn_backend import (
    ImageWAMAttnBackend,
    _fp16_tensor_from_ptr,
    make_imagewam_attention_spec,
)
from flash_rt.models.imagewam.quant_linear import Fp16Linear
from flash_rt.models.imagewam.text_context import pack_trimmed_context, trimmed_sequence_dims

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
NH, HD = 24, 128
REAL_MAX_DIMS = dict(x0=513, a0=905, total=969, num_action=64)

# Small frontend: 16 text rows + proprio, 10 image rows, 4 action rows.
TEXT_LEN, JA, PROPRIO = 16, 16, 3
SMALL_DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 10, total=TEXT_LEN + 1 + 10 + 4,
                  joint_attention_dim=JA, proprio_dim=PROPRIO)
PROPRIO_VALUE = [0.1, -0.2, 0.3]

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
# The real FA4 kernel runs only on Thor (sm_110) with the FA4 runtime installed.
REAL_FA4 = torch.cuda.is_available() and fa4_backend.fa4_fwd() is not None


def _stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af, bf = a.double().flatten(), b.double().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-30))
    return cos, float((af - bf).abs().max()), float((af - bf).norm() / (bf.norm() + 1e-30))


def _fmt(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    cos, max_abs, rel = _stats(a, b)
    return f"{name}: cosine={cos:.7f} max_abs={max_abs:.3e} rel_l2={rel:.3e}"


# ── text_context.py ─────────────────────────────────────────────────────


def _official_append_proprio():
    """Official `ImageWAM._append_proprio_to_context`, or None when the
    `imagewam` package does not import here."""
    try:
        from imagewam.models.backbones.imagewam import ImageWAM
    except Exception:  # the package pulls in flux2/diffusers; absent on some machines
        return None
    return ImageWAM._append_proprio_to_context


@pytest.mark.parametrize("valid", [
    [1] * 5 + [0] * 11,                      # prefix mask (Qwen3 right padding)
    [1, 0, 1, 1, 0, 0, 1, 0] + [0] * 8,      # non-prefix mask
])
def test_pack_matches_official_proprio_packing(valid):
    torch.manual_seed(0)
    ctx = torch.randn(TEXT_LEN, JA)
    mask = torch.tensor(valid, dtype=torch.bool)
    n = int(mask.sum())
    packed = pack_trimmed_context(ctx, mask, proprio_slot=True)
    assert packed.n_valid == n and packed.proprio_row == n and tuple(packed.rows.shape) == (n + 1, JA)
    assert torch.equal(packed.rows[:n], ctx[mask]) and bool((packed.rows[n] == 0).all())

    append = _official_append_proprio()
    if append is None:
        pytest.skip("imagewam package not importable; checked against the rank-order rule only")
    proprio_token = torch.randn(1, JA)
    official_self = SimpleNamespace(
        proprio_encoder=lambda p: proprio_token.expand(p.shape[0], p.shape[1], JA),
        proprio_dim=PROPRIO, pack_proprio_after_text=True, device="cpu")
    new_ctx, new_mask = append(official_self, ctx[None], mask[None], torch.zeros(1, PROPRIO))
    # Official: valid rows by rank, proprio at row n, mask = prefix [0, n].
    assert torch.equal(new_mask[0], torch.arange(TEXT_LEN + 1) <= n)
    assert torch.equal(new_ctx[0, :n], packed.rows[:n])
    assert torch.equal(new_ctx[0, n], proprio_token[0])
    print(f"\npack vs official _append_proprio_to_context (n_valid={n}): rows [0,{n}) equal, "
          f"proprio at row {n}, official mask is the prefix [0,{n}]")


def test_pack_without_proprio_and_errors():
    ctx = torch.randn(8, JA)
    prefix = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    packed = pack_trimmed_context(ctx, prefix, proprio_slot=False)
    assert packed.proprio_row is None and torch.equal(packed.rows, ctx[:3])
    with pytest.raises(ValueError, match="prefix"):
        pack_trimmed_context(ctx, torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool), proprio_slot=False)
    with pytest.raises(ValueError, match="no valid token"):
        pack_trimmed_context(ctx, torch.zeros(8, dtype=torch.bool), proprio_slot=True)


def test_trimmed_sequence_dims_real():
    d = dict(REAL_MAX_DIMS)
    assert trimmed_sequence_dims(d, 513) is d
    t = trimmed_sequence_dims(d, 20)
    assert (t["x0"], t["a0"], t["total"], t["num_action"]) == (20, 412, 476, 64)
    assert d["x0"] == 513, "the max dims must not be modified"
    for bad in (0, 514):
        with pytest.raises(ValueError):
            trimmed_sequence_dims(d, bad)


# ── served attention at trimmed real shapes ─────────────────────────────


def _real_backend(total_max: int, a0_max: int, *, use_fa4: bool = False, use_fa4_mot: bool = False,
                  seed: int = 0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    hidden = NH * HD
    spec = make_imagewam_attention_spec(max_prefix_seq=a0_max, max_total_seq=total_max,
                                        num_layers=1, num_heads=NH, head_dim=HD)
    q_o = torch.randn(total_max, hidden, generator=gen, device=DEV).to(FP16)
    k = torch.randn(1, total_max, hidden, generator=gen, device=DEV).to(FP16)
    v = torch.randn(1, total_max, hidden, generator=gen, device=DEV).to(FP16)
    logits = torch.zeros(total_max * NH, total_max + total_max % 2, dtype=FP16, device=DEV)
    fa4_out = torch.zeros(total_max, hidden, dtype=FP16, device=DEV)
    slots = {"Q_O": q_o.data_ptr(), "K": k.data_ptr(), "V": v.data_ptr(), "logits": logits.data_ptr(),
             "scale": HD ** -0.5, "fa4_out": fa4_out.data_ptr(), "fa4_out_numel": fa4_out.numel()}
    backend = ImageWAMAttnBackend(spec, fvk.FvkContext(), backbone_slots=dict(slots),
                                  mot_slots=dict(slots, layer_stride=k[0].numel() * 2),
                                  use_perhead_kv=True, use_real_mot_mask=True,
                                  use_fa4=use_fa4, use_fa4_mot=use_fa4_mot)
    return backend, (q_o, k, v, logits, fa4_out)


def _torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(Sq, NH*HD), (Skv, NH*HD) fp16 -> fp32 softmax attention per head."""
    qf = q.float().view(q.shape[0], NH, HD).transpose(0, 1)
    kf = k.float().view(k.shape[0], NH, HD).transpose(0, 1)
    vf = v.float().view(v.shape[0], NH, HD).transpose(0, 1)
    o = torch.softmax(qf @ kf.transpose(-1, -2) * HD ** -0.5, dim=-1) @ vf
    return o.transpose(0, 1).reshape(q.shape[0], NH * HD)


def _fa4_fp32_stand_in(q, k, v, *, causal, num_splits, pack_gqa, out, softmax_scale=None):
    """`_flash_attn_fwd` calling convention, fp32 PyTorch math."""
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    out.copy_((torch.softmax(qf @ kf.transpose(-1, -2) * scale, dim=-1) @ vf).transpose(1, 2).to(out.dtype))
    return out, None


@needs_gpu
@pytest.mark.parametrize("kernel", ["cublas_perhead", "fa4_stand_in", "fa4_real"])
@pytest.mark.parametrize("x0", [20, 21, 32, 33])
def test_served_attention_at_trimmed_real_shapes(monkeypatch, kernel, x0):
    """Trimmed LIBERO lengths: x0 = n_valid + 1 in [17, 32]; a0 = x0 + 392
    and total = a0 + 64 take both parities as x0 does. `fa4_real` runs
    the FA4 kernel itself (Thor only)."""
    if kernel == "fa4_stand_in":
        monkeypatch.setattr(fa4_backend, "fa4_fwd", lambda: _fa4_fp32_stand_in)
    if kernel == "fa4_real" and not REAL_FA4:
        pytest.skip(f"no FA4 runtime: {fa4_backend.status()}")
    t = trimmed_sequence_dims(dict(REAL_MAX_DIMS), x0)
    a0, total = t["a0"], t["total"]
    fa4 = kernel != "cublas_perhead"
    backend, (q_o, k, v, _, _) = _real_backend(REAL_MAX_DIMS["total"], REAL_MAX_DIMS["a0"],
                                               use_fa4=fa4, use_fa4_mot=fa4)
    q_in = q_o.clone()
    backend.run("backbone", 0, q_seq=a0, kv_seq=a0, stream=0)
    backend.run("mot", 0, q_seq=64, kv_seq=total, stream=0, x0=x0, a0=a0)
    torch.cuda.synchronize()
    ref_backbone = _torch_attention(q_in[:a0], k[0, :a0], v[0, :a0])
    ref_mot = _torch_attention(q_in[a0:total], k[0, :total], v[0, :total])
    print(f"\n{kernel} x0={x0} a0={a0} total={total}: "
          + _fmt("backbone", q_o[:a0], ref_backbone) + " | " + _fmt("mot", q_o[a0:total], ref_mot))
    assert _stats(q_o[:a0], ref_backbone)[0] > 0.9999
    assert _stats(q_o[a0:total], ref_mot)[0] > 0.9999
    assert torch.equal(q_o[total:], q_in[total:]), "rows past the trimmed sequence must be untouched"


# ── small-dims frontend ──────────────────────────────────────────────────


class _TorchReferenceAttn(ImageWAMAttnBackend):
    """fp32 PyTorch attention on the frontend's own buffers, both sites,
    optionally excluding key columns `[lo, hi)` for every query (the
    official text-padding mask). Capturable."""

    masked_keys: tuple[int, int] | None = None

    def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0, state_nk=None, x0=None, a0=None):
        spec = self._spec.site(site)
        nh, hd = spec.num_q_heads, spec.head_dim
        kv_seq = q_seq if kv_seq is None else kv_seq
        slot = self._slots[site]
        k_ptr, v_ptr = self._per_layer_kv[layer_idx]
        q_ptr = int(slot["Q_O"]) + (0 if site == "backbone" else int(a0) * nh * hd * 2)
        q = _fp16_tensor_from_ptr(q_ptr, (q_seq, nh, hd))
        k = _fp16_tensor_from_ptr(k_ptr, (kv_seq, nh, hd))
        v = _fp16_tensor_from_ptr(v_ptr, (kv_seq, nh, hd))
        ctx = self._fa4_stream_context(stream) if stream else nullcontext()
        with ctx:
            logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(slot["scale"])
            if self.masked_keys is not None:
                logits[:, :, self.masked_keys[0]:self.masked_keys[1]] = float("-inf")
            out = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), v.float())
            q.copy_(out.to(FP16))
        return q_ptr


def _reference_attn(fe: ImageWAMTorchFrontendThor, masked_keys: tuple[int, int] | None) -> _TorchReferenceAttn:
    attn = _TorchReferenceAttn(fe._attn_spec, fe._ctx, backbone_slots=dict(fe._attn_slots["backbone"]),
                               mot_slots=dict(fe._attn_slots["mot"]), use_perhead_kv=True,
                               use_real_mot_mask=True)
    attn.masked_keys = masked_keys
    return attn


def _frontend(*, text_trim: bool, gemm_runner=None, seed: int = 0, **kw) -> ImageWAMTorchFrontendThor:
    torch.manual_seed(seed)  # identical random weights for every frontend built with the same seed
    return ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(SMALL_DIMS), text_trim=text_trim,
                                     gemm_runner=gemm_runner, **kw)


def _context(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TEXT_LEN, JA, generator=g).to(BF16)


def _mask(n_valid: int) -> torch.Tensor:
    m = torch.zeros(TEXT_LEN, dtype=torch.bool)
    m[:n_valid] = True
    return m


_NOISE = torch.randn(4, 7, generator=torch.Generator().manual_seed(2))


def _run(fe: ImageWAMTorchFrontendThor, ctx: torch.Tensor, n_valid: int) -> torch.Tensor:
    fe.set_prompt(context=ctx, context_mask=_mask(n_valid))
    torch.manual_seed(7)  # infer() draws the (placeholder) image tokens from the CUDA RNG
    out = fe.infer({"proprio": PROPRIO_VALUE}, action_noise=_NOISE.to(DEV))["actions"]
    return torch.from_numpy(out).double()


@needs_gpu
@pytest.mark.parametrize("n_valid", [5, 8])
def test_trimmed_equals_untrimmed_with_masked_text_keys(n_valid):
    ctx = _context()
    masked = _frontend(text_trim=False)
    gemm = masked._gemm
    masked._attn = _reference_attn(masked, masked_keys=(n_valid + 1, SMALL_DIMS["x0"]))
    ref = _run(masked, ctx, n_valid)

    trimmed_ref_attn = _frontend(text_trim=True, gemm_runner=gemm)
    trimmed_ref_attn._attn = _reference_attn(trimmed_ref_attn, masked_keys=None)
    trimmed_served = _frontend(text_trim=True, gemm_runner=gemm)
    unmasked_ref_attn = _frontend(text_trim=False, gemm_runner=gemm)
    unmasked_ref_attn._attn = _reference_attn(unmasked_ref_attn, masked_keys=None)
    unmasked_served = _frontend(text_trim=False, gemm_runner=gemm)

    out = {
        "trimmed, fp32 attention": _run(trimmed_ref_attn, ctx, n_valid),
        "trimmed, served attention": _run(trimmed_served, ctx, n_valid),
        "untrimmed unmasked, fp32 attention": _run(unmasked_ref_attn, ctx, n_valid),
        "untrimmed unmasked, served attention": _run(unmasked_served, ctx, n_valid),
    }
    print(f"\nn_valid={n_valid}, x0 {SMALL_DIMS['x0']} -> {n_valid + 1}; reference: untrimmed with keys "
          f"[{n_valid + 1}, {SMALL_DIMS['x0']}) masked, fp32 attention")
    for name, value in out.items():
        print("  " + _fmt(f"{name:38s} vs reference", value, ref))
    assert trimmed_ref_attn.active_dims["x0"] == n_valid + 1
    exact = _stats(out["trimmed, fp32 attention"], ref)
    served = _stats(out["trimmed, served attention"], ref)
    unmasked = _stats(out["untrimmed unmasked, fp32 attention"], ref)
    assert exact[2] < 1e-5
    assert served[0] > 0.9999
    assert unmasked[2] > 1e-4, "the padded keys must change the untrimmed, unmasked result at these dims"


@needs_gpu
def test_switching_lengths_and_back_matches_fresh_frontends():
    ctx = _context()
    fe = _frontend(text_trim=True)
    a1 = _run(fe, ctx, 5)
    graph_a = fe._graph
    b = _run(fe, ctx, 9)
    full = _run(fe, ctx, TEXT_LEN)
    a2 = _run(fe, ctx, 5)
    print(f"\ncaptured lengths {fe.captured_text_lengths}; " + _fmt("A after B and full vs A", a2, a1)
          + " | " + _fmt("A vs B", a1, b))
    assert fe.captured_text_lengths == (6, 10, TEXT_LEN + 1)
    assert fe._graph is graph_a, "a cached length must replay its own graph, not recapture"
    assert torch.equal(a2, a1)
    assert fe._active_dims is not fe.dims and fe.active_dims["a0"] == 6 + 10
    for n, value in ((9, b), (TEXT_LEN, full)):
        fresh = _frontend(text_trim=True, gemm_runner=fe._gemm)
        other = _run(fresh, ctx, n)
        print("  " + _fmt(f"length {n + 1} after switching vs a fresh frontend", value, other))
        assert torch.equal(value, other)
    assert fe.captured_text_lengths[-1] == fe.dims["x0"] and fe._captures[fe.dims["x0"]].dims is fe.dims


@needs_gpu
@pytest.mark.parametrize("text_trim", [False, True])
def test_new_context_of_the_same_length_is_applied(text_trim):
    """issues.md ISSUE-060: a second, different context with the same mask."""
    ctx_a, ctx_b = _context(1), _context(3)
    fe = _frontend(text_trim=text_trim)
    out_a = _run(fe, ctx_a, 5)
    out_b = _run(fe, ctx_b, 5)
    fresh_b = _run(_frontend(text_trim=text_trim, gemm_runner=fe._gemm), ctx_b, 5)
    print(f"\ntext_trim={text_trim}: " + _fmt("second context vs first", out_b, out_a)
          + " | second context vs a fresh frontend: equal=" + str(torch.equal(out_b, fresh_b)))
    assert torch.equal(fe._context[:5].cpu(), ctx_b[:5])
    assert not torch.equal(out_b, out_a)
    assert torch.equal(out_b, fresh_b)
    assert len(fe.captured_text_lengths) == 1


@needs_gpu
def test_untrimmed_keeps_one_capture_at_the_max_length():
    fe = _frontend(text_trim=False)
    for n in (5, 9, TEXT_LEN):
        _run(fe, _context(), n)
        assert fe._active_dims is fe.dims and fe._rope_table is fe._max_rope_table
    assert fe.captured_text_lengths == (SMALL_DIMS["x0"],)


@needs_gpu
def test_live_text_encoder_path_trims_and_caches_by_prompt(monkeypatch):
    """`set_prompt(prompt_text)` with a text encoder: the returned mask sets
    the length; the same prompt again returns early (no re-encode)."""
    import flash_rt.models.imagewam.text_encoder as text_encoder

    lengths = {"short prompt": 4, "longer prompt": 11}
    calls: list[str] = []

    def encode(model, tokenizer, prompts):
        calls.append(prompts[0])
        return _context()[None].to(DEV), _mask(lengths[prompts[0]])[None].to(DEV)

    monkeypatch.setattr(text_encoder, "encode_prompts", encode)
    fe = _frontend(text_trim=True)
    fe._qwen3 = (None, None)
    for prompt in ("short prompt", "short prompt", "longer prompt", "short prompt"):
        fe.set_prompt(prompt)
        assert fe.active_dims["x0"] == lengths[prompt] + 1 and fe._proprio_row == lengths[prompt]
    print(f"\nencoder calls {calls}; captured lengths {fe.captured_text_lengths}")
    assert calls == ["short prompt", "longer prompt", "short prompt"]
    assert fe.captured_text_lengths == (5, 12)


@needs_gpu
def test_random_prompt_path_uses_the_max_length():
    fe = _frontend(text_trim=True)
    fe.set_prompt("random context")
    assert fe.active_dims["x0"] == SMALL_DIMS["x0"] and fe._proprio_row == SMALL_DIMS["x0"] - 1


class _GrowingScratchLinear(Fp16Linear):
    """`Fp16Linear` that stages its input through a scratch buffer grown on
    demand, the allocation pattern of `Nvfp4Linear` / `Fp8Linear` (quantize
    into scratch, GEMM from scratch). Records every allocation size."""

    def __init__(self, gemm, weight_ptr: int, n: int, k: int):
        super().__init__(gemm, weight_ptr, n, k)
        self.scratch = None
        self.allocations: list[int] = []

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if self.scratch is None or m > self.scratch.shape[0]:
            self.scratch = torch.empty(m, self.k, dtype=FP16, device=DEV)
            self.allocations.append(m)
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)) if stream else nullcontext():
            self.scratch[:m].copy_(_fp16_tensor_from_ptr(x_ptr, (m, self.k)))
        self.gemm.fp16_nn(self.scratch.data_ptr(), self.weight_ptr, out_ptr, m, self.n, self.k, stream)


@needs_gpu
@pytest.mark.parametrize("reserve", [True, False])
def test_lazily_grown_scratch_is_sized_once_at_the_max(monkeypatch, reserve):
    """With the max-dims prefill (`reserve=True`, the frontend's behavior)
    every scratch is allocated once, at its largest `m`; without it
    (`reserve=False`, control) a longer length reallocates scratch that
    earlier graphs still read."""
    monkeypatch.setattr(ImageWAMTorchFrontendThor, "_wrap_linear",
                        lambda self, w, n, k: _GrowingScratchLinear(self._gemm, w.data_ptr(), n, k))
    fe = _frontend(text_trim=True)
    if not reserve:
        fe._scratch_reserved = True
    ctx = _context()
    first = _run(fe, ctx, 4)
    _run(fe, ctx, 12)
    again = _run(fe, ctx, 4)
    ops = {id(op): (key, op) for key, op in fe._weights.items() if isinstance(op, _GrowingScratchLinear)}
    by_site: dict[str, set[tuple[int, ...]]] = {}
    for key, op in ops.values():
        by_site.setdefault(key[0], set()).add(tuple(op.allocations))
    print(f"\nreserve={reserve}: allocation sizes per op, backbone {sorted(by_site['backbone'])}, "
          f"action_dit {sorted(by_site['action_dit'])}; " + _fmt("length 5 again vs first", again, first))
    x0, a0, na = SMALL_DIMS["x0"], SMALL_DIMS["a0"], 4
    if reserve:
        assert by_site["backbone"] <= {(x0,), (a0,), (a0 - x0,)}
        assert by_site["action_dit"] == {(na,)}
        assert torch.equal(again, first)
    else:
        assert any(len(sizes) > 1 for sizes in by_site["backbone"]), "control: a longer length must reallocate"


@needs_gpu
@pytest.mark.parametrize("fa4", ["stand_in", "real"])
def test_trimmed_with_fa4_matches_cublas(monkeypatch, fa4):
    """FA4 at both sites (`use_fa4`, `use_fa4_mot`) at trimmed lengths:
    `stand_in` replaces FA4 with fp32 PyTorch math (dispatch only);
    `real` runs the FA4 kernel inside the captured graphs (Thor only)."""
    if fa4 == "stand_in":
        monkeypatch.setattr(fa4_backend, "fa4_fwd", lambda: _fa4_fp32_stand_in)
    elif not REAL_FA4:
        pytest.skip(f"no FA4 runtime: {fa4_backend.status()}")
    ctx = _context()
    cublas = _frontend(text_trim=True)
    fa4 = _frontend(text_trim=True, gemm_runner=cublas._gemm, use_fa4=True, use_fa4_mot=True)
    for n in (5, 6, 12):
        ref, out = _run(cublas, ctx, n), _run(fa4, ctx, n)
        print(f"\n  x0={n + 1} " + _fmt(f"FA4 ({fa4}) vs cuBLAS chain", out, ref))
        assert _stats(out, ref)[0] > 0.9999
    assert fa4.fa4_fallback_reason is None and fa4.captured_text_lengths == (6, 7, 13)


@needs_gpu
def test_too_many_valid_tokens_is_rejected():
    fe = _frontend(text_trim=True)
    ctx = torch.randn(TEXT_LEN + 1, JA).to(BF16)
    with pytest.raises(ValueError, match="more than"):
        fe.set_prompt(context=ctx, context_mask=torch.ones(TEXT_LEN + 1, dtype=torch.bool))


_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH") or os.environ.get("FLUX2_AE_MODEL_PATH", "")
_AE_AVAILABLE = os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)
# Image span of the real 2x224x224 VAE output (392 tokens, 14x28 grid).
_VAE_DIMS = dict(SMALL_DIMS, a0=SMALL_DIMS["x0"] + 392, total=SMALL_DIMS["x0"] + 392 + 4, ref_h=14, ref_w=28)


@needs_gpu
@pytest.mark.skipif(not _AE_AVAILABLE, reason="real flux2 clone / AE checkpoint not present")
def test_vae_in_graph_shares_one_pool_across_lengths():
    """`vae_graph_input` + `text_trim`: every length's graph records the VAE
    stage, all in one shared graph pool. Switching lengths and back gives
    the actions of the VAE-outside-the-graph frontend, bit for bit."""
    g = torch.Generator().manual_seed(4)
    views = {"view1": torch.randint(0, 256, (224, 224, 3), generator=g, dtype=torch.uint8),
             "view2": torch.randint(0, 256, (224, 224, 3), generator=g, dtype=torch.uint8),
             "proprio": PROPRIO_VALUE}
    ctx = _context()

    def build(**kw) -> ImageWAMTorchFrontendThor:
        torch.manual_seed(0)
        return ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(_VAE_DIMS), text_trim=True,
                                         ae_model_path=_AE_PATH, flux2_src=_FLUX2_SRC, **kw)

    def run(fe: ImageWAMTorchFrontendThor, n_valid: int) -> torch.Tensor:
        fe.set_prompt(context=ctx, context_mask=_mask(n_valid))
        return torch.from_numpy(fe.infer(dict(views), action_noise=_NOISE.to(DEV))["actions"]).double()

    in_graph = build(vae_graph_input=(2, 224, 224))
    outside = build(gemm_runner=in_graph._gemm)
    for n in (5, 12, 5, 9, 12):
        a, b = run(in_graph, n), run(outside, n)
        print(f"\n  x0={n + 1} " + _fmt("VAE in graph vs VAE outside", a, b))
        assert torch.equal(a, b)
    assert in_graph.captured_text_lengths == (6, 10, 13)

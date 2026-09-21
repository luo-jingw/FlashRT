"""Multi-length CUDA-graph and buffer safety of `text_trim=True` at any
precision (issues.md ISSUE-080, opportunities.md OPT-030).

One frontend serves a sequence of prompt lengths (A, a longer B, a
shorter C, A again, the maximum, ...), one CUDA graph per length over the
same max-size buffers, all captured into one graph pool. Checked here:

- every replay of a length equals that length's first replay, and equals
  a fresh frontend that only ever saw that length (same seeded weights,
  shared `GemmRunner`), bit for bit;
- no weight op reallocates a tensor it owns (activation scratch,
  quantized weights) after the first capture: the device pointers of
  every tensor reachable from every op are unchanged after all lengths;
- replays still match after the graph pool's free blocks and the regular
  caching allocator's free memory are filled with 0xFF, and no replay
  writes into that memory (memory the frontend no longer owns);
- with the VAE stage inside every length's graph (needs the real AE);
- a capture that raises leaves no graph active, and the frontend
  recovers.

The precision comes from the environment, so on Thor this covers the
quantized weight ops (NVFP4, FP8, E0M3) and FA4 across lengths; the other
frontend-level trim tests run fp16 only.

Environment:

    TRIM_PRECISION   fp16 (default), fp16_cutlass, fp8, fp8_static,
                     fp8_static_cutlass, nvfp4, e0m3_hadamard
    TRIM_FA4         off (default) | on: FA4 at both attention sites
                     (use_fa4, use_fa4_mot); skipped without an FA4 runtime
    TRIM_DIMS        small (default) | real: the real FLUX.2-4B / ActionDiT
                     dims (x0=513 max, a0=905, 64 actions, 10 steps)
    TRIM_WEIGHTS     random (default) | real: the checkpoint at CKPT_PATH
                     (needs TRIM_DIMS=real)
    TRIM_CALIBRATION activation-calibration file (`calibration_path=`) for
                     fp8_static / fp8_static_cutlass; needs TRIM_WEIGHTS=real
                     and a file recorded with text_trim (builder
                     `--text-trim`). Without it the static FP8 scales are
                     the frontend's placeholder calibration.
    TRIM_POISON_MIB  size of the regular-cache poison, default 1024

A precision this machine cannot run (NVFP4 / E0M3 / SM100 CUTLASS without
an sm_110 build, or FP8 cuBLASLt in a layout this GPU does not support: the
frontend picks NN on compute capability >= 10 and TN below,
`quant_linear.fp8_cublaslt_layout()`) skips
with the reason. Example (Thor):

    TRIM_PRECISION=nvfp4 TRIM_FA4=on python -m pytest \\
        tests/test_imagewam_text_trim_graph_safety.py -q -s
"""
from __future__ import annotations

import gc
import os

import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.hardware.thor import fa4_backend

DEV = "cuda"
BF16 = torch.bfloat16
PRECISION = os.environ.get("TRIM_PRECISION", "fp16")
FA4 = os.environ.get("TRIM_FA4", "off") == "on"
DIMS_NAME = os.environ.get("TRIM_DIMS", "small")
POISON_MIB = int(os.environ.get("TRIM_POISON_MIB", "1024"))
REAL_WEIGHTS = os.environ.get("TRIM_WEIGHTS", "random") == "real"
CALIBRATION = os.environ.get("TRIM_CALIBRATION") or None
if REAL_WEIGHTS and DIMS_NAME != "real":
    raise ValueError("TRIM_WEIGHTS=real needs TRIM_DIMS=real")
if CALIBRATION is not None and not REAL_WEIGHTS:
    raise ValueError("TRIM_CALIBRATION needs TRIM_WEIGHTS=real (a calibration file belongs to one checkpoint)")

if DIMS_NAME == "real":
    TEXT_LEN = 512
    DIMS = dict(
        hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
        x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 392, num_layers_double=5, num_layers_single=20,
        action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
        num_action=64, total=TEXT_LEN + 1 + 392 + 64,
        action_num_layers_double=5, action_num_layers_single=20,
        dt=0.1, num_denoise_steps=10, ref_h=14, ref_w=28, proprio_dim=8, shift=5.0,
        num_train_timesteps=1000)
    SEQUENCE = (20, 31, 16, 20, TEXT_LEN, 16, 31, 20, TEXT_LEN, 25)
else:
    TEXT_LEN = 16
    # 392 image tokens (the real 2x224x224 VAE grid) so the VAE check can
    # use the same dims; every GEMM K and N is a multiple of 16.
    DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 392, total=TEXT_LEN + 1 + 392 + 4,
                joint_attention_dim=16, proprio_dim=3, ref_h=14, ref_w=28)
    SEQUENCE = (5, 12, 3, 5, TEXT_LEN, 3, 12, 5, TEXT_LEN, 9)
PROPRIO = [0.1 * (i + 1) for i in range(DIMS["proprio_dim"])]
NUM_ACTION = DIMS.get("num_action", 4)

_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH") or os.environ.get("FLUX2_AE_MODEL_PATH", "")
_AE_AVAILABLE = os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Errors that mean "this machine cannot run the precision / FA4", not a bug.
_ENVIRONMENT_GAPS = ("Blackwell/Thor", "cublasLtMatmulAlgoGetHeuristic failed", "cutlass_fp16_",
                     "cutlass_fp8_", "active Thor FA4 runtime")


def _environment_gap(exc: BaseException) -> str | None:
    text = f"{type(exc).__name__}: {exc}"
    return text if any(gap in text for gap in _ENVIRONMENT_GAPS) else None


def _stats(a: torch.Tensor, b: torch.Tensor) -> str:
    af, bf = a.double().flatten(), b.double().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-30))
    return f"equal={torch.equal(a, b)} cosine={cos:.9f} max_abs={float((af - bf).abs().max()):.3e}"


_CONTEXT = torch.randn(TEXT_LEN, DIMS["joint_attention_dim"], generator=torch.Generator().manual_seed(1)).to(BF16)
_NOISE = torch.randn(NUM_ACTION, 7, generator=torch.Generator().manual_seed(2))
_VIEWS = {name: torch.randint(0, 256, (224, 224, 3), generator=torch.Generator().manual_seed(seed),
                              dtype=torch.uint8) for name, seed in (("view1", 3), ("view2", 4))}


def _mask(n_valid: int) -> torch.Tensor:
    m = torch.zeros(TEXT_LEN, dtype=torch.bool)
    m[:n_valid] = True
    return m


def _build(gemm_runner=None, *, vae: bool = False,
           fa4: bool | None = None) -> ImageWAMTorchFrontendThor:
    """A frontend for this module's configuration. `fa4` overrides the
    environment's `TRIM_FA4`, for a reference frontend that has to capture
    on the same attention chain as another one (see the recovery test)."""
    use_fa4 = FA4 if fa4 is None else fa4
    if use_fa4 and fa4_backend.fa4_fwd() is None:
        pytest.skip(f"TRIM_FA4=on needs an FA4 runtime: {fa4_backend.status()}")
    kw: dict[str, object] = {}
    if vae:
        kw.update(ae_model_path=_AE_PATH, flux2_src=_FLUX2_SRC, vae_graph_input=(2, 224, 224))
    if REAL_WEIGHTS:
        kw.update(ckpt_path=os.environ["CKPT_PATH"])
    if CALIBRATION is not None:
        kw.update(calibration_path=CALIBRATION)
    torch.manual_seed(0)  # the same random weights for every frontend
    try:
        return ImageWAMTorchFrontendThor(precision=PRECISION, dims_override=dict(DIMS), text_trim=True,
                                         gemm_runner=gemm_runner, use_fa4=use_fa4, use_fa4_mot=use_fa4, **kw)
    except (RuntimeError, AttributeError) as exc:
        reason = _environment_gap(exc)
        if reason is None:
            raise
        pytest.skip(f"precision {PRECISION} does not run here: {reason}")


def _run(fe: ImageWAMTorchFrontendThor, n_valid: int, *, vae: bool = False) -> torch.Tensor:
    torch.manual_seed(11)  # static-FP8 placeholder calibration draws at the first capture
    try:
        fe.set_prompt(context=_CONTEXT, context_mask=_mask(n_valid))
    except RuntimeError as exc:
        reason = _environment_gap(exc)
        if reason is None:
            raise
        pytest.skip(f"precision {PRECISION} does not run here: {reason}")
    obs: dict[str, object] = {"proprio": PROPRIO}
    if vae:
        obs.update(_VIEWS)
    torch.manual_seed(7)  # placeholder image tokens without a VAE
    return torch.from_numpy(fe.infer(obs, action_noise=_NOISE.to(DEV))["actions"]).double()


def _owned_tensor_ptrs(fe: ImageWAMTorchFrontendThor) -> dict[str, int]:
    """Device pointer of every CUDA tensor reachable from each weight op:
    its attributes, dicts and lists, and one level of nested FlashRT
    objects (for example an `FP4ActScratch`)."""
    out: dict[str, int] = {}

    def visit(prefix: str, value: object, depth: int) -> None:
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                out[prefix] = value.data_ptr()
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(f"{prefix}[{k}]", v, depth)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(f"{prefix}[{i}]", v, depth)
        elif depth < 2 and type(value).__module__.startswith("flash_rt.") and not callable(value):
            attrs = getattr(value, "__dict__", None)
            for k, v in (attrs.items() if isinstance(attrs, dict) else ()):
                visit(f"{prefix}.{k}", v, depth + 1)

    seen: set[int] = set()
    for key, op in fe._weights.items():
        if isinstance(op, int) or id(op) in seen:
            continue
        seen.add(id(op))
        for k, v in vars(op).items():
            visit(f"{'.'.join(map(str, key))}.{k}", v, 1)
    return out


def _fresh(n_valid: int, gemm_runner, *, vae: bool = False, fa4: bool | None = None) -> torch.Tensor:
    fresh = _build(gemm_runner, vae=vae, fa4=fa4)
    out = _run(fresh, n_valid, vae=vae)
    assert fresh.captured_text_lengths == (n_valid + 1,)
    del fresh
    gc.collect()
    return out


def _free_pool_block_sizes(pool) -> list[int]:
    sizes: list[int] = []
    for seg in torch.cuda.memory_snapshot():
        if tuple(seg.get("segment_pool_id", (0, 0))) == tuple(pool):
            sizes += [blk["size"] for blk in seg["blocks"] if blk["state"] == "inactive"]
    return sizes


def _pool_poison_graph(fe: ImageWAMTorchFrontendThor) -> tuple[torch.cuda.CUDAGraph, int]:
    """A graph in the frontend's graph pool that fills every free block of
    that pool with 0xFF when replayed."""
    sizes = _free_pool_block_sizes(fe._graph_pool)
    graph = torch.cuda.CUDAGraph()
    s = fe._graph_stream
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, pool=fe._graph_pool, stream=s):
        held = [torch.empty(size, dtype=torch.uint8, device=DEV) for size in sizes]
        for t in held:
            t.fill_(0xFF)
        del held
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return graph, sum(sizes)


def _poison_regular_cache(fe: ImageWAMTorchFrontendThor) -> list[torch.Tensor]:
    """Release the caching allocator's free memory, then take `POISON_MIB`
    back, filled with 0xFF, on the default and the capture stream (a freed
    block is reused only on the stream it was freed on)."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    held = []
    per_stream = POISON_MIB // 2
    for stream in (torch.cuda.current_stream(), fe._graph_stream):
        with torch.cuda.stream(stream):
            sizes = [64 << 20] * (per_stream // 64) + [512 * k for k in range(1, 513)]
            for size in sizes:
                t = torch.empty(size, dtype=torch.uint8, device=DEV)
                t.fill_(0xFF)
                held.append(t)
    torch.cuda.synchronize()
    return held


def _check_lengths(vae: bool) -> None:
    fe = _build(vae=vae)
    first: dict[int, torch.Tensor] = {}
    ptrs_after_first = None
    for n in SEQUENCE:
        out = _run(fe, n, vae=vae)
        if ptrs_after_first is None:
            ptrs_after_first = _owned_tensor_ptrs(fe)
        if n in first:
            assert torch.equal(out, first[n]), f"n_valid={n}: replay differs from its first replay: {_stats(out, first[n])}"
        else:
            first[n] = out
            assert torch.isfinite(out).all(), f"n_valid={n}: non-finite actions"
    print(f"\nprecision={PRECISION} fa4={FA4} dims={DIMS_NAME} weights={'real' if REAL_WEIGHTS else 'random'} "
          f"calibration={CALIBRATION} vae={vae}: sequence {SEQUENCE}, "
          f"cached x0 {fe.captured_text_lengths}, fa4_fallback_reason={fe.fa4_fallback_reason!r}")

    ptrs = _owned_tensor_ptrs(fe)
    moved = sorted(k for k in ptrs_after_first if ptrs.get(k) != ptrs_after_first[k])
    added = sorted(set(ptrs) - set(ptrs_after_first))
    print(f"  weight-op tensors: {len(ptrs)}; moved after the first capture: {len(moved)}; "
          f"added after it: {len(added)}")
    assert not moved and not added, f"weight-op tensors reallocated after the first capture: {(moved + added)[:8]}"

    for n in sorted(first):
        fresh = _fresh(n, fe._gemm, vae=vae)
        print(f"  n_valid={n:3d} x0={n + 1:3d}: after switching vs fresh single-length frontend {_stats(first[n], fresh)}")
        assert torch.equal(first[n], fresh)

    poison, pool_bytes = _pool_poison_graph(fe)
    order = sorted(first) + sorted(first, reverse=True)
    for n in order:
        poison.replay()
        out = _run(fe, n, vae=vae)
        assert torch.equal(out, first[n]), f"n_valid={n} after graph-pool poison: {_stats(out, first[n])}"
    held = _poison_regular_cache(fe)
    for n in order:
        out = _run(fe, n, vae=vae)
        assert torch.equal(out, first[n]), f"n_valid={n} after regular-cache poison: {_stats(out, first[n])}"
    torch.cuda.synchronize()
    clobbered = sum(int((t != 0xFF).sum().item()) for t in held)
    print(f"  graph-pool poison {pool_bytes / 2**20:.1f} MiB, regular-cache poison "
          f"{sum(t.numel() for t in held) / 2**20:.0f} MiB: {2 * len(order)} replays all equal; "
          f"poison bytes overwritten by replays: {clobbered}")
    assert clobbered == 0, "a replay wrote into memory the frontend no longer owns"


def test_lengths_switch_safely():
    _check_lengths(vae=False)


@pytest.mark.skipif(not _AE_AVAILABLE, reason="real flux2 clone / AE checkpoint not present")
def test_lengths_switch_safely_with_the_vae_in_every_graph():
    _check_lengths(vae=True)


def test_capture_failure_leaves_no_graph_and_recovers(monkeypatch):
    lengths = sorted(set(SEQUENCE))[:3]
    fe = _build()
    first = _run(fe, lengths[0])
    assert fe._graph is not None and fe.captured_text_lengths == (lengths[0] + 1,)
    real_capture = fe._capture_graph

    def boom() -> None:
        raise RuntimeError("stand-in: capture failed on a new length")

    monkeypatch.setattr(fe, "_capture_graph", boom)
    with pytest.raises(RuntimeError, match="stand-in"):
        fe.set_prompt(context=_CONTEXT, context_mask=_mask(lengths[1]))
    assert fe._graph is None and fe._current_prompt is None
    with pytest.raises(RuntimeError, match="set_prompt"):
        fe.infer({"proprio": PROPRIO})
    # With FA4 on, `_capture_graph_or_fall_back` reads ANY capture failure as an
    # FA4 failure, records it and switches this frontend to the cuBLAS chain for
    # good -- its documented contract, since FA4 is the compiled site. With FA4
    # off there is no fallback and the stand-in's error just propagates. Either
    # way the frontend now serves the cuBLAS chain (`use_fa4` off).
    assert (fe.fa4_fallback_reason is not None) == FA4
    assert fe.use_fa4 is False and fe.use_fa4_mot is False
    if FA4:
        assert "stand-in" in fe.fa4_fallback_reason
    monkeypatch.setattr(fe, "_capture_graph", real_capture)
    recovered = _run(fe, lengths[0])
    if not FA4:
        # The failure never consulted the fallback: `lengths[0]` replay its own
        # cached graph, on the chain both were captured on, bit for bit.
        assert torch.equal(recovered, first), f"n_valid={lengths[0]}: {_stats(recovered, first)}"
    else:
        # `first` is an FA4 capture and the re-capture runs the cuBLAS chain the
        # fallback switched to, so the two differ in the last bits by
        # construction -- comparing them would measure FA4 against cuBLAS, not
        # the recovery (the dispatch tests measure that same gap). Like for
        # like, the reference is a frontend that captured on the chain this one
        # now serves.
        chain_first = _fresh(lengths[0], fe._gemm, fa4=False)
        assert torch.equal(recovered, chain_first), f"n_valid={lengths[0]}: {_stats(recovered, chain_first)}"
    out = _run(fe, lengths[1])
    fresh = _fresh(lengths[1], fe._gemm, fa4=False)
    print(f"\nprecision={PRECISION} fa4={FA4}: after a failed capture, n_valid={lengths[1]} vs fresh {_stats(out, fresh)}")
    assert torch.equal(out, fresh)


def test_tensor_census_detects_a_reallocation(monkeypatch):
    """Control for the pointer census above: an op that grows its scratch
    on a larger `m` (the quantized ops' pattern), with the max-dims
    prefill skipped, must show up as moved. Runs fp16 stand-in ops at any
    `TRIM_PRECISION`."""
    from flash_rt.models.imagewam.quant_linear import Fp16Linear

    class GrowingScratchLinear(Fp16Linear):
        def __init__(self, gemm, weight_ptr: int, n: int, k: int):
            super().__init__(gemm, weight_ptr, n, k)
            self.scratch = None

        def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
            if self.scratch is None or m > self.scratch.shape[0]:
                self.scratch = torch.empty(m, self.k, dtype=torch.float16, device=DEV)
            self.gemm.fp16_nn(x_ptr, self.weight_ptr, out_ptr, m, self.n, self.k, stream)

    monkeypatch.setattr(ImageWAMTorchFrontendThor, "_wrap_linear",
                        lambda self, w, n, k: GrowingScratchLinear(self._gemm, w.data_ptr(), n, k))
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(DIMS), text_trim=True)
    fe._scratch_reserved = True  # skip the max-dims prefill
    lengths = sorted(set(SEQUENCE))
    _run(fe, lengths[0])
    before = _owned_tensor_ptrs(fe)
    _run(fe, lengths[-1])
    after = _owned_tensor_ptrs(fe)
    moved = [k for k in before if after.get(k) != before[k]]
    print(f"\ncontrol without the max-dims prefill: {len(moved)} of {len(before)} scratch tensors moved")
    assert moved

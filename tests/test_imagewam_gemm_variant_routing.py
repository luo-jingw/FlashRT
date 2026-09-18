"""Tile-variant routing for the CUTLASS-backed ImageWAM linears and the
frontend's ActionDiT tuning pass (plan.md, "Plan: ActionDiT small-M
CUTLASS tile selection", roadmap item 1).

The NVFP4 and SM100 FP8 CUTLASS kernels cannot run on sm_90. These tests
replace only the GEMM entry points with recorders (`flash_rt_fp4` as a
whole module; the `cutlass_fp8_*` attributes of `flash_rt_kernels`) and
keep everything else real: `Nvfp4Linear`, `StaticFp8Linear`, the
frontend's weight construction and grouping, `GemmVariantTuner`, and the
CUDA graph capture in `set_prompt()`. The recorders write a constant into
the GEMM output so the tuner's correctness gate and the captured graph
see finite data. A stub timer makes a chosen variant win, so the tests
observe where the choice lands, not how fast anything is.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch

import flash_rt
import flash_rt.flash_rt_kernels as fvk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

FP16 = torch.float16


@dataclass(frozen=True)
class _GemmCall:
    variant: str
    m: int
    n: int
    k: int


def _fill_out(out_ptr: int, m: int, n: int) -> None:
    interface = {"data": (int(out_ptr), False), "shape": (m, n), "typestr": "<f2", "version": 3}
    owner = type("_Out", (), {"__cuda_array_interface__": interface})()
    torch.as_tensor(owner, device="cuda").fill_(1.0)


def _fake_fp4_module(calls: list[_GemmCall]) -> types.ModuleType:
    mod = types.ModuleType("flash_rt.flash_rt_fp4")

    def sfa_size_bytes(rows: int, d: int, is_sfb: bool) -> int:
        return ((rows + 127) // 128) * 128 * (d // 16)

    def quantize_fp4_dynamic_sfa_fp16(x, packed, sfa, m, k, is_sfb, stream) -> int:
        return 0

    def cutlass_fp4_gemm_variant(idx, a, sfa, b, sfb, d, m, n, k, alpha=1.0, beta=0.0, stream=0) -> int:
        calls.append(_GemmCall(f"v{idx}", m, n, k))
        _fill_out(d, m, n)
        return 0

    def cutlass_fp4_gemm_num_variants() -> int:
        return 11

    mod.sfa_size_bytes = sfa_size_bytes
    mod.quantize_fp4_dynamic_sfa_fp16 = quantize_fp4_dynamic_sfa_fp16
    mod.cutlass_fp4_gemm_variant = cutlass_fp4_gemm_variant
    mod.cutlass_fp4_gemm_num_variants = cutlass_fp4_gemm_num_variants
    return mod


@pytest.fixture
def fake_fp4(monkeypatch):
    calls: list[_GemmCall] = []
    mod = _fake_fp4_module(calls)
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", mod)
    monkeypatch.setattr(flash_rt, "flash_rt_fp4", mod, raising=False)
    monkeypatch.delitem(sys.modules, "flash_rt.executors.fp4_utils", raising=False)
    yield calls
    sys.modules.pop("flash_rt.executors.fp4_utils", None)


@pytest.fixture
def fake_fp8_cutlass(monkeypatch):
    from flash_rt.models.imagewam.quant_linear import FP8_CUTLASS_VARIANTS

    calls: list[_GemmCall] = []

    def make(variant: str):
        def fn(a, b, d, m, n, k, alpha=1.0, beta=0.0, stream=0) -> int:
            calls.append(_GemmCall(variant, m, n, k))
            _fill_out(d, m, n)
            return 0
        return fn

    for v in FP8_CUTLASS_VARIANTS:
        monkeypatch.setattr(fvk, f"cutlass_fp8_{v}", make(v), raising=False)
    return calls


class _PreferTimer:
    """Stub `VariantTimer`: runs each batch once and reports `fast_us` for
    the variant `prefer` and `slow_us` for every other one."""

    def __init__(self, calls: list[_GemmCall], prefer: str, fast_us: float = 1.0, slow_us: float = 5.0):
        self._calls = calls
        self._prefer = prefer
        self._fast = fast_us
        self._slow = slow_us
        self.batches_timed = 0

    def us_per_launch(self, batches, launches_per_batch: int) -> tuple[float, ...]:
        out = []
        for batch in batches:
            batch(0)
            self.batches_timed += 1
            out.append(self._fast if self._calls[-1].variant == self._prefer else self._slow)
        return tuple(out)


def _weight(k: int, n: int) -> torch.Tensor:
    return torch.randn(k, n, dtype=FP16, device="cuda") * 0.02


def test_nvfp4_linear_default_matches_pick_variant_and_switches(fake_fp4):
    from flash_rt.executors.fp4_utils import pick_variant
    from flash_rt.models.imagewam.quant_linear import NVFP4_VARIANTS, Nvfp4Linear

    for n, k in ((9216, 1024), (1024, 3072), (17408, 1024)):
        w = _weight(k, n)
        lin = Nvfp4Linear(w.data_ptr(), n, k)
        assert lin.family == "nvfp4"
        assert lin.default_variant == f"v{pick_variant(n, k)}" == lin.variant
        assert lin.default_variant in NVFP4_VARIANTS
        x = torch.randn(64, k, dtype=FP16, device="cuda")
        out = torch.empty(64, n, dtype=FP16, device="cuda")
        lin(x.data_ptr(), out.data_ptr(), 64, 0)
        lin.set_variant("v10")
        lin(x.data_ptr(), out.data_ptr(), 64, 0)
        assert fake_fp4[-2:] == [_GemmCall(lin.default_variant, 64, n, k), _GemmCall("v10", 64, n, k)]
    with pytest.raises(ValueError):
        lin.set_variant("v11")


def test_static_fp8_cutlass_default_matches_heuristic_and_switches(fake_fp8_cutlass):
    from flash_rt.models.imagewam.quant_linear import StaticFp8Linear, _pick_fp8_cutlass_variant

    n, k = 1024, 4096
    w = _weight(k, n)
    lin = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=True)
    assert lin.family == "fp8_cutlass"
    assert lin.default_variant == _pick_fp8_cutlass_variant(n, k) == lin.variant == "sq"
    x = torch.randn(64, k, dtype=FP16, device="cuda")
    out = torch.empty(64, n, dtype=FP16, device="cuda")
    lin.prepare_tuning_input(x.data_ptr(), 64, 0)
    assert lin.launch_variant("t128x64x256", out.data_ptr(), 64, 0) == 0
    # Tuning must leave the calibrate-before-call contract untouched.
    lin.calibrate(x.data_ptr(), 64, 0)
    lin.set_variant("t128x64x256")
    lin(x.data_ptr(), out.data_ptr(), 64, 0)
    assert fake_fp8_cutlass == [_GemmCall("t128x64x256", 64, n, k), _GemmCall("t128x64x256", 64, n, k)]
    cublaslt = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
    assert cublaslt.family == "fp8_cublaslt" and cublaslt.variant == "cublaslt"
    with pytest.raises(RuntimeError):
        cublaslt.candidate_variants()


_SMALL_DIMS = dict(num_action=16, total=24)  # total = a0 (8) + num_action


def _action_shapes(d: dict) -> set[tuple[int, int]]:
    ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
    shapes = {(3 * aaw, ahd), (ahd, aaw), (2 * amh, ahd), (ahd, amh), (3 * aaw + 2 * amh, ahd)}
    if d["merge_linear2"]:
        shapes.add((ahd, aaw + amh))  # single-stream merged linear2
    return shapes


def _build(monkeypatch, precision: str, calls: list[_GemmCall], prefer: str):
    import flash_rt.frontends.torch.imagewam_thor as imagewam_thor

    timers: list[_PreferTimer] = []

    def make_timer() -> _PreferTimer:
        timers.append(_PreferTimer(calls, prefer))
        return timers[-1]

    monkeypatch.setattr(imagewam_thor, "CudaGraphVariantTimer", make_timer)
    fe = imagewam_thor.ImageWAMTorchFrontendThor(
        precision=precision, dims_override=dict(_SMALL_DIMS), gemm_variant_autotune=True)
    return fe, timers


@pytest.mark.parametrize("precision,prefer", [("nvfp4", "v10"), ("fp8_static_cutlass", "t128x64x256")])
def test_frontend_tunes_each_action_dit_shape_once(monkeypatch, fake_fp4, fake_fp8_cutlass, precision, prefer):
    from flash_rt.models.imagewam.quant_linear import Nvfp4Linear, StaticFp8Linear

    calls = fake_fp4 if precision == "nvfp4" else fake_fp8_cutlass
    fe, timers = _build(monkeypatch, precision, calls, prefer)
    d = fe.dims
    results = fe.gemm_variant_results
    for r in results:
        print(r.summary())
    assert {(r.shape.n, r.shape.k) for r in results} == _action_shapes(d)
    assert all(r.shape.m == d["num_action"] and r.chosen_variant == prefer for r in results)
    assert timers[0].batches_timed == sum(len(r.measurements) for r in results)
    members = {(r.shape.n, r.shape.k): r.members for r in results}
    n_double, n_single = d["action_num_layers_double"], d["action_num_layers_single"]
    ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
    if d["merge_linear2"]:
        assert members[(ahd, aaw)] == n_double                 # proj
        assert members[(ahd, amh)] == n_double                 # mlp2
        assert members[(ahd, aaw + amh)] == n_single           # merged linear2
    else:
        assert members[(ahd, aaw)] == n_double + n_single      # proj + attn_out_proj
        assert members[(ahd, amh)] == n_double + n_single      # mlp2 + mlp_down

    cls = Nvfp4Linear if precision == "nvfp4" else StaticFp8Linear
    for key, lin in fe._weights.items():
        if not isinstance(lin, cls):
            continue
        expected = prefer if key[0] == "action_dit" else lin.default_variant
        assert lin.variant == expected, key

    # The captured graph runs the chosen tile for every ActionDiT GEMM and
    # the untouched heuristic tile for the backbone.
    calls.clear()
    fe.set_prompt("routing")
    captured = {c for c in calls}
    action_calls = {c for c in captured if c.m == d["num_action"]}
    backbone_calls = captured - action_calls
    assert {c.variant for c in action_calls} == {prefer}
    assert {(c.n, c.k) for c in action_calls} == _action_shapes(d)
    assert prefer not in {c.variant for c in backbone_calls}
    actions = fe.infer({})["actions"]
    assert np.isfinite(actions).all()


def test_flag_rejected_for_untunable_precision():
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    with pytest.raises(ValueError, match="gemm_variant_autotune"):
        ImageWAMTorchFrontendThor(precision="fp16", gemm_variant_autotune=True)


def test_flag_off_leaves_heuristic_tiles(monkeypatch, fake_fp4):
    import flash_rt.frontends.torch.imagewam_thor as imagewam_thor
    from flash_rt.models.imagewam.quant_linear import Nvfp4Linear

    fe = imagewam_thor.ImageWAMTorchFrontendThor(precision="nvfp4", dims_override=dict(_SMALL_DIMS))
    assert fe.gemm_variant_results == ()
    lins = [lin for lin in fe._weights.values() if isinstance(lin, Nvfp4Linear)]
    assert lins and all(lin.variant == lin.default_variant for lin in lins)
    assert fake_fp4 == []


def test_stale_build_without_small_m_fp8_tiles_still_constructs(monkeypatch, fake_fp8_cutlass):
    """A `flash_rt_kernels` built before the 1-SM FP8 tiles existed lacks the
    `cutlass_fp8_t128x*` symbols: those candidates are rejected with the
    AttributeError, and tuning chooses among the tiles that exist."""
    for v in ("t128x64x256", "t128x64x128", "t128x128x128", "t128x256x128"):
        monkeypatch.delattr(fvk, f"cutlass_fp8_{v}", raising=False)
    fe, _ = _build(monkeypatch, "fp8_static_cutlass", fake_fp8_cutlass, prefer="plain")
    for r in fe.gemm_variant_results:
        print(r.summary())
        by_v = {x.variant: x for x in r.measurements}
        assert all(by_v[v].status.startswith("launch_failed AttributeError")
                   for v in by_v if v.startswith("t128x"))
        assert r.chosen_variant == "plain"

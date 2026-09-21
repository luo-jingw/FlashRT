"""Measured per-shape GEMM tile-variant selection (plan.md, "Plan:
ActionDiT small-M CUTLASS tile selection", roadmap item 1).

The kernels this selects between (SM100 CUTLASS FP8, NVFP4) cannot run
on sm_90, so the selection rule is tested against stub GEMMs on the
CPU: each stub computes `x @ W` in PyTorch and writes the result through
the raw output pointer, and a stub timer reports a fixed time per
variant. The last test runs the real CUDA-graph timer over two real
sm_90 launch batches and prints what it measured.
"""
from __future__ import annotations

import ctypes

import pytest
import torch

from flash_rt.models.imagewam.gemm_variant_tuner import (
    STATUS_MISMATCH,
    STATUS_NONFINITE,
    STATUS_OK,
    GemmVariantTuner,
    VariantTunableGemm,
)

FP16 = torch.float16


class _StubGemm:
    """CPU `VariantTunableGemm`. `rcs[v]` makes variant `v` return that
    code, `perturb[v]` adds noise of that scale to its output, and
    variants in `nonfinite` write an inf."""

    def __init__(self, *, seed: int, family: str = "stub", n: int = 24, k: int = 32,
                 default: str = "a", candidates: tuple[str, ...] = ("a", "b", "c"),
                 rcs: dict[str, int] | None = None, perturb: dict[str, float] | None = None,
                 nonfinite: frozenset[str] = frozenset(), raises: dict[str, Exception] | None = None):
        self.family, self.n, self.k = family, n, k
        self.default_variant = default
        self.variant = default
        self._candidates = candidates
        self._rcs = rcs or {}
        self._perturb = perturb or {}
        self._nonfinite = nonfinite
        self._raises = raises or {}
        gen = torch.Generator().manual_seed(seed)
        self._w = torch.randn(k, n, generator=gen)
        self._x: torch.Tensor | None = None
        self.launch_log: list[str] = []

    def candidate_variants(self) -> tuple[str, ...]:
        return self._candidates

    def set_variant(self, variant: str) -> None:
        self.variant = variant

    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int) -> None:
        x = torch.empty(m, self.k, dtype=FP16)
        ctypes.memmove(x.data_ptr(), x_ptr, x.numel() * x.element_size())
        self._x = x

    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int) -> int:
        self.launch_log.append(variant)
        if variant in self._raises:
            raise self._raises[variant]
        rc = self._rcs.get(variant, 0)
        if rc != 0:
            return rc
        assert self._x is not None
        y = self._x.float() @ self._w
        if variant in self._perturb:
            y = y + self._perturb[variant] * torch.randn(y.shape, generator=torch.Generator().manual_seed(7))
        if variant in self._nonfinite:
            y[0, 0] = float("inf")
        y16 = y.to(FP16).contiguous()
        ctypes.memmove(out_ptr, y16.data_ptr(), y16.numel() * y16.element_size())
        return 0


class _StubTimer:
    """Runs each batch once and reports the fixed time of the variant the
    batch launched (read back from the first member's launch log);
    variants listed in `untimeable` come back as `None`, as from a batch
    that could not be captured."""

    def __init__(self, members: list[_StubGemm], us: dict[str, float],
                 untimeable: frozenset[str] = frozenset()):
        self._members = members
        self._us = us
        self._untimeable = untimeable
        self.calls = 0

    def us_per_launch(self, batches, launches_per_batch: int) -> tuple[float | None, ...]:
        self.calls += 1
        out = []
        for batch in batches:
            before = sum(len(m.launch_log) for m in self._members)
            batch(0)
            launched = sum(len(m.launch_log) for m in self._members) - before
            assert launched == launches_per_batch
            v = self._members[0].launch_log[-1]
            out.append(None if v in self._untimeable else self._us[v])
        return tuple(out)


def _tune(members: list[_StubGemm], us: dict[str, float], m: int = 4,
          untimeable: frozenset[str] = frozenset(), **kw):
    timer = _StubTimer(members, us, untimeable)
    tuner = GemmVariantTuner(timer, device="cpu", **kw)
    result = tuner.tune(members, m)
    return tuner, timer, result


def _group(n_members: int = 3, **kw) -> list[_StubGemm]:
    return [_StubGemm(seed=i, **kw) for i in range(n_members)]


def test_stub_satisfies_protocol():
    stub: VariantTunableGemm = _StubGemm(seed=0)
    assert stub.default_variant == "a"


def test_picks_fastest_and_applies_to_every_member():
    members = _group()
    _, _, result = _tune(members, {"a": 10.0, "b": 5.0, "c": 8.0})
    print(result.summary())
    assert result.chosen_variant == "b"
    assert all(m.variant == "b" for m in members)
    assert [x.status for x in result.measurements] == [STATUS_OK] * 3
    assert result.members == 3
    assert result.speedup_vs_default() == pytest.approx(2.0)


def test_keeps_default_within_min_gain():
    members = _group()
    _, _, result = _tune(members, {"a": 10.0, "b": 9.9, "c": 12.0}, min_gain=0.02)
    assert result.chosen_variant == "a"
    assert all(m.variant == "a" for m in members)
    assert result.speedup_vs_default() == 1.0


def test_launch_failure_is_rejected_not_timed():
    members = _group(rcs={"b": 0x10002})
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0})
    by_v = {x.variant: x for x in result.measurements}
    print(result.summary())
    assert by_v["b"].status.startswith("launch_failed rc=")
    assert by_v["b"].us_per_gemm is None
    assert result.chosen_variant == "c"


def test_mismatching_output_is_rejected():
    members = _group(perturb={"b": 5.0})
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0})
    by_v = {x.variant: x for x in result.measurements}
    print(f"mismatch cosine={by_v['b'].cosine_vs_default:.6f}, c cosine={by_v['c'].cosine_vs_default:.6f}")
    assert by_v["b"].status == STATUS_MISMATCH
    assert by_v["c"].cosine_vs_default == pytest.approx(1.0, abs=1e-6)
    assert result.chosen_variant == "c"


def test_nonfinite_output_is_rejected():
    members = _group(nonfinite=frozenset({"b"}))
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0})
    by_v = {x.variant: x for x in result.measurements}
    assert by_v["b"].status == STATUS_NONFINITE
    assert result.chosen_variant == "c"


def test_failure_on_one_member_rejects_the_candidate():
    members = _group()
    members[2]._rcs = {"b": 3}
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0})
    assert result.chosen_variant == "c"


def test_candidate_raising_is_rejected_not_fatal():
    """A stale build without a candidate's kernel symbol raises
    AttributeError from `getattr(fvk, ...)`; the candidate is rejected and
    tuning completes."""
    members = _group(raises={"b": AttributeError("module 'flash_rt_kernels' has no attribute "
                                                 "'cutlass_fp8_t128x64x256'")})
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0})
    by_v = {x.variant: x for x in result.measurements}
    print(result.summary())
    assert by_v["b"].status.startswith("launch_failed AttributeError:")
    assert by_v["b"].us_per_gemm is None
    assert result.chosen_variant == "c"


def test_default_raising_is_an_error():
    members = _group(raises={"a": AttributeError("missing default kernel")})
    with pytest.raises(RuntimeError, match="default variant 'a' raised AttributeError"):
        _tune(members, {"a": 10.0, "b": 5.0, "c": 8.0})


def test_untimeable_candidate_is_rejected():
    members = _group()
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0}, untimeable=frozenset({"b"}))
    by_v = {x.variant: x for x in result.measurements}
    assert by_v["b"].status == "timing_failed" and by_v["b"].us_per_gemm is None
    assert result.chosen_variant == "c"


def test_untimeable_default_is_kept():
    members = _group()
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 8.0}, untimeable=frozenset({"a"}))
    assert result.chosen_variant == "a"
    assert all(m.variant == "a" for m in members)


def test_default_failure_raises():
    members = _group(rcs={"a": 7})
    with pytest.raises(RuntimeError, match="default variant 'a' failed rc=7"):
        _tune(members, {"a": 10.0, "b": 5.0, "c": 8.0})


def test_every_other_candidate_rejected_keeps_default():
    members = _group(rcs={"b": 1, "c": 2})
    _, _, result = _tune(members, {"a": 10.0, "b": 1.0, "c": 1.0})
    assert result.chosen_variant == "a"
    assert result.us_of("a") == 10.0


def test_default_is_considered_even_if_not_listed():
    members = _group(default="z", candidates=("b", "c"))
    _, _, result = _tune(members, {"z": 3.0, "b": 5.0, "c": 8.0})
    assert [x.variant for x in result.measurements] == ["z", "b", "c"]
    assert result.chosen_variant == "z"


def test_cache_applies_choice_without_relaunching():
    members = _group()
    tuner, timer, first = _tune(members, {"a": 10.0, "b": 5.0, "c": 8.0})
    later = _group(n_members=2)
    second = tuner.tune(later, 4)
    assert second is first
    assert all(m.variant == "b" for m in later)
    assert all(m.launch_log == [] for m in later)
    assert timer.calls == 1
    assert tuner.results() == (first,)


def test_different_m_is_a_different_cache_entry():
    members = _group()
    timer = _StubTimer(members, {"a": 10.0, "b": 5.0, "c": 8.0})
    tuner = GemmVariantTuner(timer, device="cpu")
    tuner.tune(members, 4)
    tuner.tune(members, 8)
    assert [r.shape.m for r in tuner.results()] == [4, 8]


def test_members_with_different_shapes_raise():
    members = [_StubGemm(seed=0), _StubGemm(seed=1, n=16)]
    tuner = GemmVariantTuner(_StubTimer(members, {}), device="cpu")
    with pytest.raises(ValueError, match="must share family/N/K/default"):
        tuner.tune(members, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_graph_timer_on_real_launches():
    """Two real launch batches on any CUDA GPU: `x1` runs one cuBLASLt
    fp16 GEMM per member, `x4` runs the same GEMM four times. Prints the
    per-launch times the CUDA-graph timer reports next to a direct
    CUDA-event measurement of eager launches."""
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.models.imagewam.gemm_variant_timer import CudaGraphVariantTimer

    m, n, k, members = 64, 1024, 3072, 6
    gemm = fvk.GemmRunner()
    weights = [torch.randn(k, n, dtype=FP16, device="cuda") * 0.02 for _ in range(members)]
    x = torch.randn(m, k, dtype=FP16, device="cuda")
    out = torch.empty(m, n, dtype=FP16, device="cuda")
    gemm.fp16_nn(x.data_ptr(), weights[0].data_ptr(), out.data_ptr(), m, n, k, 0)
    torch.cuda.synchronize()

    def batch(times: int):
        def run(stream: int) -> None:
            for w in weights:
                for _ in range(times):
                    gemm.fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, stream)
        return run

    timer = CudaGraphVariantTimer(reps=4, samples=15, warmup=3)
    us_x1, us_x4 = timer.us_per_launch([batch(1), batch(4)], members)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    eager = []
    for _ in range(15):
        start.record()
        batch(1)(torch.cuda.current_stream().cuda_stream)
        end.record()
        end.synchronize()
        eager.append(start.elapsed_time(end) * 1000.0 / members)
    eager.sort()
    print(f"\ngraph-timed x1={us_x1:.2f}us x4={us_x4:.2f}us (ratio {us_x4 / us_x1:.2f}); "
          f"eager event-timed x1 P50={eager[len(eager) // 2]:.2f}us (M={m} N={n} K={k}, shared GPU)")
    assert us_x1 > 0.0 and us_x4 > 0.0
    assert all(t == t and t != float("inf") for t in (us_x1, us_x4))

    def raising(stream: int) -> None:
        raise AttributeError("stand-in: kernel symbol missing")

    def invalidating(stream: int) -> None:
        batch(1)(stream)
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()  # not permitted during capture: invalidates it

    caller = torch.cuda.current_stream()
    times = timer.us_per_launch([raising, batch(1), invalidating], members)
    print(f"batch that raises -> {times[0]}, good batch -> {times[1]:.2f}us, "
          f"capture-invalidating batch -> {times[2]}")
    # The invalidating batch leaves the default CUDA generator believing a capture is still open on
    # torch 2.9.1, and every later random draw in the process then fails (ISSUE-087): one successful
    # small capture clears it, so this test does not poison the tests after it.
    reset = torch.cuda.CUDAGraph()
    probe = torch.zeros(1, device="cuda")
    with torch.cuda.graph(reset):
        _ = probe + 1
    assert times[0] is None and times[2] is None and times[1] > 0.0
    assert torch.cuda.current_stream() == caller

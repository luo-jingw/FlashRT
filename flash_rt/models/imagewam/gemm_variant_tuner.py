"""Measured per-shape tile-variant selection for ImageWAM's quantized GEMMs.

A CUTLASS GEMM family (NVFP4 block-scaled, FP8 SM100) ships several tile
and cluster configurations. Which one is fastest depends on the problem
shape and the device, and a static `(N, K)` table tuned at one `M` does
not transfer to another (opportunities.md OPT-014, result 3: at the
ActionDiT's `M = 64`, the FP8 CUTLASS pick ran 1.44-1.68x slower than
cuBLASLt). This module replaces the guess with a one-time measurement on
the device the frontend runs on, per group of linears that share
`(family, M, N, K)`.

Selection rule for one group (`GemmVariantTuner.tune`):

1. The same random `(M, K)` fp16 input is staged into every member
   (`prepare_tuning_input`).
2. Correctness gate, eager: every member runs its default variant into
   a reference buffer, then every other candidate into a second buffer.
   A candidate is rejected on a nonzero return code, a non-finite
   output, or a cosine against the default's output below
   `cosine_floor` on any member. All variants of one family compute the
   same math on the same quantized operands, so a correct candidate
   differs from the default only by fp32 accumulation order.
3. Timing: each surviving candidate is timed as one launch per member,
   round robin, so consecutive launches read different layers' weights
   (the deployed pipeline reads each weight once per step, never from a
   warm L2). Launch overhead is excluded by the timer (see
   `gemm_variant_timer.py`).
4. The fastest candidate wins only if it beats the default by more than
   `min_gain`; otherwise the default is kept.
5. The choice is applied to every member (`set_variant`) and cached per
   `(family, M, N, K)`; a later `tune` call with the same key applies
   the cached choice without re-measuring.

The default variant failing its own launch is an error, not a case to
route around: that linear would fail in the served pipeline too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch

FP16 = torch.float16

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_NONFINITE = "nonfinite"
STATUS_LAUNCH_FAILED = "launch_failed"


@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class VariantMeasurement:
    """One candidate's outcome. `status` is `ok`, `mismatch`, `nonfinite`,
    or `launch_failed rc=<code>`. `cosine_vs_default` is the minimum over
    the group's members (1.0 for the default itself); `us_per_gemm` is
    the timed cost of one GEMM launch, `None` for a rejected candidate."""
    variant: str
    status: str
    cosine_vs_default: float | None
    us_per_gemm: float | None


@dataclass(frozen=True)
class VariantTuneResult:
    family: str
    shape: GemmShape
    members: int
    default_variant: str
    chosen_variant: str
    measurements: tuple[VariantMeasurement, ...]

    def us_of(self, variant: str) -> float | None:
        for meas in self.measurements:
            if meas.variant == variant:
                return meas.us_per_gemm
        return None

    def speedup_vs_default(self) -> float:
        """Default time over chosen time from this measurement (1.0 when
        the default is kept)."""
        default_us = self.us_of(self.default_variant)
        chosen_us = self.us_of(self.chosen_variant)
        if default_us is None or chosen_us is None or chosen_us <= 0.0:
            return 1.0
        return default_us / chosen_us

    def summary(self) -> str:
        parts = []
        for meas in self.measurements:
            if meas.us_per_gemm is None:
                parts.append(f"{meas.variant}:{meas.status}")
            else:
                parts.append(f"{meas.variant}:{meas.us_per_gemm:.2f}us")
        s = self.shape
        return (f"{self.family} M={s.m} N={s.n} K={s.k} x{self.members}: "
                f"default={self.default_variant} chosen={self.chosen_variant} "
                f"({self.speedup_vs_default():.3f}x) [{' '.join(parts)}]")


class VariantTunableGemm(Protocol):
    """A linear op whose GEMM kernel variant can be switched at runtime.

    `launch_variant` runs only the GEMM, on operands staged by the last
    `prepare_tuning_input` call, into `out_ptr` (`(m, n)` fp16), and
    returns the kernel's return code (0 = success). It never changes the
    op's own state. `set_variant` selects the variant `__call__` uses.
    """
    family: str
    n: int
    k: int
    default_variant: str
    variant: str

    def candidate_variants(self) -> tuple[str, ...]: ...

    def set_variant(self, variant: str) -> None: ...

    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int) -> None: ...

    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int) -> int: ...


class VariantTimer(Protocol):
    """Times several launch batches, interleaved. Each batch issues
    `launches_per_batch` GEMM launches on the stream it is given. Returns
    the per-launch time in microseconds for each batch, in order."""

    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float, ...]: ...


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    if float(denom) == 0.0:
        return 1.0 if bool(torch.equal(af, bf)) else 0.0
    return float((af @ bf) / denom)


class GemmVariantTuner:
    """Owns the selection rule and the `(family, M, N, K)` result cache
    for one frontend. See the module docstring for the rule."""

    def __init__(self, timer: VariantTimer, *, device: str = "cuda",
                 min_gain: float = 0.02, cosine_floor: float = 0.9999, seed: int = 0):
        if not 0.0 <= min_gain < 1.0:
            raise ValueError(f"min_gain must be in [0, 1), got {min_gain}")
        self._timer = timer
        self._device = device
        self._min_gain = float(min_gain)
        self._cosine_floor = float(cosine_floor)
        self._seed = int(seed)
        self._cache: dict[tuple[str, int, int, int], VariantTuneResult] = {}
        self._order: list[tuple[str, int, int, int]] = []

    def results(self) -> tuple[VariantTuneResult, ...]:
        return tuple(self._cache[key] for key in self._order)

    def tune(self, members: Sequence[VariantTunableGemm], m: int) -> VariantTuneResult:
        if len(members) == 0:
            raise ValueError("tune() needs at least one member")
        first = members[0]
        family, n, k, default = first.family, int(first.n), int(first.k), first.default_variant
        candidates = self._candidates(first)
        for member in members[1:]:
            if (member.family, int(member.n), int(member.k), member.default_variant) != (family, n, k, default):
                raise ValueError(
                    f"tune() members must share family/N/K/default: {(family, n, k, default)} vs "
                    f"{(member.family, member.n, member.k, member.default_variant)}")
            if self._candidates(member) != candidates:
                raise ValueError("tune() members must share one candidate set")
        m = int(m)
        key = (family, m, n, k)
        cached = self._cache.get(key)
        if cached is not None:
            for member in members:
                member.set_variant(cached.chosen_variant)
            return cached

        statuses, cosines = self._check_candidates(members, candidates, m, n, k, default)
        survivors = [v for v in candidates if statuses[v] == STATUS_OK]
        times = self._time_candidates(members, survivors, m, n)

        default_us = times[default]
        best = min(survivors, key=lambda v: times[v])
        chosen = default
        if best != default and times[best] < default_us * (1.0 - self._min_gain):
            chosen = best

        measurements = tuple(
            VariantMeasurement(variant=v, status=statuses[v], cosine_vs_default=cosines[v],
                               us_per_gemm=times.get(v))
            for v in candidates)
        result = VariantTuneResult(family=family, shape=GemmShape(m, n, k), members=len(members),
                                   default_variant=default, chosen_variant=chosen,
                                   measurements=measurements)
        for member in members:
            member.set_variant(chosen)
        self._cache[key] = result
        self._order.append(key)
        return result

    @staticmethod
    def _candidates(member: VariantTunableGemm) -> tuple[str, ...]:
        """Default first, then the member's other candidates in its own
        order, without duplicates."""
        ordered = [member.default_variant]
        for v in member.candidate_variants():
            if v not in ordered:
                ordered.append(v)
        return tuple(ordered)

    def _check_candidates(self, members: Sequence[VariantTunableGemm], candidates: tuple[str, ...],
                          m: int, n: int, k: int, default: str
                          ) -> tuple[dict[str, str], dict[str, float | None]]:
        gen = torch.Generator(device=self._device).manual_seed(self._seed)
        x = torch.randn(m, k, generator=gen, dtype=torch.float32, device=self._device).to(FP16)
        for member in members:
            member.prepare_tuning_input(x.data_ptr(), m, 0)
        self._sync()
        ref = torch.empty(m, n, dtype=FP16, device=self._device)
        out = torch.empty(m, n, dtype=FP16, device=self._device)
        statuses: dict[str, str] = {v: STATUS_OK for v in candidates}
        cosines: dict[str, float | None] = {v: None for v in candidates}
        cosines[default] = 1.0
        for idx, member in enumerate(members):
            rc = member.launch_variant(default, ref.data_ptr(), m, 0)
            self._sync()
            if rc != 0:
                raise RuntimeError(
                    f"{member.family} default variant {default!r} failed rc={rc} "
                    f"(M={m}, N={n}, K={k}, member {idx}) -- the served path would fail too")
            if not bool(torch.isfinite(ref).all()):
                raise RuntimeError(
                    f"{member.family} default variant {default!r} produced non-finite output "
                    f"(M={m}, N={n}, K={k}, member {idx})")
            for v in candidates:
                if v == default or statuses[v] != STATUS_OK:
                    continue
                out.fill_(0)
                rc = member.launch_variant(v, out.data_ptr(), m, 0)
                self._sync()
                if rc != 0:
                    statuses[v] = f"{STATUS_LAUNCH_FAILED} rc={rc}"
                    continue
                if not bool(torch.isfinite(out).all()):
                    statuses[v] = STATUS_NONFINITE
                    continue
                cos = _cosine(out, ref)
                prev = cosines[v]
                cosines[v] = cos if prev is None else min(prev, cos)
                if cos < self._cosine_floor:
                    statuses[v] = STATUS_MISMATCH
        return statuses, cosines

    def _time_candidates(self, members: Sequence[VariantTunableGemm], survivors: list[str],
                         m: int, n: int) -> dict[str, float]:
        out = torch.empty(m, n, dtype=FP16, device=self._device)
        out_ptr = out.data_ptr()

        def make_batch(variant: str) -> Callable[[int], None]:
            def batch(stream: int) -> None:
                for member in members:
                    member.launch_variant(variant, out_ptr, m, stream)
            return batch

        batches = [make_batch(v) for v in survivors]
        per_launch = self._timer.us_per_launch(batches, len(members))
        if len(per_launch) != len(survivors):
            raise RuntimeError(f"timer returned {len(per_launch)} times for {len(survivors)} batches")
        del out
        return {v: float(t) for v, t in zip(survivors, per_launch)}

    def _sync(self) -> None:
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()


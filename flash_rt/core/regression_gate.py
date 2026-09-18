"""Fidelity and latency regression-gate policy, independent of any model.

A gate run measures one configuration (model, precision, device) and
compares it with fixed limits:

* Fidelity: per-sample cosine similarity against a reference, summarized
  as median and minimum over the fixture, each with its own lower bound;
  mean absolute error against ground truth, bounded by a ratio of the
  reference's own error; every output finite. Cosines themselves come
  from ``flash_rt.core.parity.parity_metrics``; this module only judges
  them.
* Latency: ``p50 < baseline_p50 * (1 + margin)``, from a per-device
  policy. Latency is ungated when the device is marked ungated (for
  example a shared GPU whose timings are contaminated by another
  tenant), when the device matches no policy, or when a gated device has
  no baseline for the precision. An ungated latency is never silent: the
  report's top-level ``latency`` field says ``ungated``, ``latency_reason``
  says why, and the verdict reason repeats it. With
  ``require_latency=True`` an ungated latency makes the verdict
  ``blocked``.

The report (``GateReport.to_dict``, schema ``RESULT_SCHEMA_VERSION``)
has the same shape as Pi0.5's end-to-end harness result
(``tests/bench_pi05_decoder_fp4_e2e.py``): named checks, a context
record (commit, device, clock state, configuration), and one verdict.

Verdicts: ``pass`` (every gated check passed; an ungated latency is
named in the reason unless it is required), ``fail`` (a check failed),
``skipped`` (the configuration is not gateable yet, e.g. a required
calibration file is absent), ``blocked`` (the configuration should be
gateable but a required interface, baseline or input is missing,
including an ungated latency under ``require_latency``). The process
exit code is 0 for ``pass``/``skipped`` and 1 for ``fail``/``blocked``.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RESULT_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
LATENCY_GROUP_COUNT = 10

CHECK_PASS = "pass"
CHECK_FAIL = "fail"
CHECK_UNGATED = "ungated"
LATENCY_CHECK = "latency_p50"
LATENCY_NOT_MEASURED = "not_measured"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_SKIPPED = "skipped"
VERDICT_BLOCKED = "blocked"


@dataclass(frozen=True)
class CosineSummary:
    """Per-sample cosines of one comparison, summarized."""

    median: float
    minimum: float
    mean: float
    count: int

    @classmethod
    def from_values(cls, values: Sequence[float]) -> CosineSummary:
        if not values:
            raise ValueError("cosine summary needs at least one value")
        array = np.asarray(values, dtype=np.float64)
        return cls(median=float(np.median(array)), minimum=float(array.min()),
                   mean=float(array.mean()), count=int(array.size))


@dataclass(frozen=True)
class GateCheck:
    """One named check. ``value``/``limit`` are the measured and allowed values."""

    name: str
    status: str
    value: float | bool | None
    limit: float | bool | None
    detail: str


@dataclass(frozen=True)
class FidelityThresholds:
    """Lower bounds on cosines and an upper bound on the MAE ratio."""

    vs_official_median_min: float
    vs_official_min_min: float
    vs_fp16_reference_median_min: float
    vs_fp16_reference_min_min: float
    mae_vs_gt_ratio_max: float
    requires_calibration: bool
    note: str

    @classmethod
    def from_json(cls, record: Mapping[str, object]) -> FidelityThresholds:
        return cls(
            vs_official_median_min=float(record["vs_official_median_min"]),
            vs_official_min_min=float(record["vs_official_min_min"]),
            vs_fp16_reference_median_min=float(record["vs_fp16_reference_median_min"]),
            vs_fp16_reference_min_min=float(record["vs_fp16_reference_min_min"]),
            mae_vs_gt_ratio_max=float(record["mae_vs_gt_ratio_max"]),
            requires_calibration=bool(record.get("requires_calibration", False)),
            note=str(record.get("note", "")))


@dataclass(frozen=True)
class FidelityThresholdTable:
    """Per-precision fidelity thresholds, loaded from a JSON config file."""

    space: str
    precisions: dict[str, FidelityThresholds]

    @classmethod
    def load(cls, path: Path) -> FidelityThresholdTable:
        record = json.loads(Path(path).read_text())
        if record.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {record.get('schema_version')} != {CONFIG_SCHEMA_VERSION}")
        return cls(space=str(record["space"]),
                   precisions={name: FidelityThresholds.from_json(value)
                               for name, value in record["precisions"].items()})

    def for_precision(self, precision: str) -> FidelityThresholds | None:
        return self.precisions.get(precision)


@dataclass(frozen=True)
class FidelityMeasurement:
    """What one gate run measured for fidelity."""

    vs_official: CosineSummary
    vs_fp16_reference: CosineSummary
    mae_vs_gt_mean: float
    reference_mae_vs_gt_mean: float
    all_finite: bool


class FidelityGate:
    """Judges a ``FidelityMeasurement`` against ``FidelityThresholds``."""

    def __init__(self, thresholds: FidelityThresholds) -> None:
        self.thresholds = thresholds

    def evaluate(self, measured: FidelityMeasurement) -> list[GateCheck]:
        t = self.thresholds
        mae_limit = measured.reference_mae_vs_gt_mean * t.mae_vs_gt_ratio_max
        return [
            _check("outputs_finite", measured.all_finite, measured.all_finite, True,
                   "every action value is finite"),
            _at_least("vs_official_median", measured.vs_official.median, t.vs_official_median_min),
            _at_least("vs_official_min", measured.vs_official.minimum, t.vs_official_min_min),
            _at_least("vs_fp16_reference_median", measured.vs_fp16_reference.median,
                      t.vs_fp16_reference_median_min),
            _at_least("vs_fp16_reference_min", measured.vs_fp16_reference.minimum,
                      t.vs_fp16_reference_min_min),
            _check("mae_vs_gt_mean", measured.mae_vs_gt_mean <= mae_limit, measured.mae_vs_gt_mean,
                   mae_limit, f"<= fp16 reference MAE {measured.reference_mae_vs_gt_mean:.5f}"
                              f" x {t.mae_vs_gt_ratio_max}"),
        ]


@dataclass(frozen=True)
class LatencySummary:
    """Latency distribution of one timed loop, in milliseconds."""

    iters: int
    p10_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    group_medians_ms: tuple[float, ...]

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> LatencySummary:
        """Group medians keep run order, so drift across the loop stays visible."""
        if len(samples_ms) < LATENCY_GROUP_COUNT:
            raise ValueError(f"latency summary requires at least {LATENCY_GROUP_COUNT} samples")
        array = np.asarray(samples_ms, dtype=np.float64)
        groups = tuple(float(np.median(g)) for g in np.array_split(array, LATENCY_GROUP_COUNT))
        return cls(iters=int(array.size), p10_ms=float(np.percentile(array, 10)),
                   p50_ms=float(statistics.median(array.tolist())),
                   p90_ms=float(np.percentile(array, 90)), min_ms=float(array.min()),
                   max_ms=float(array.max()), mean_ms=float(array.mean()), group_medians_ms=groups)


@dataclass(frozen=True)
class LatencyBaseline:
    """A recorded P50 and the allowed relative regression."""

    p50_ms: float
    margin: float
    source: str

    @property
    def limit_ms(self) -> float:
        return self.p50_ms * (1.0 + self.margin)


@dataclass(frozen=True)
class DeviceLatencyPolicy:
    """Whether latency is gated on one device, and its per-precision baselines."""

    device: str
    gated: bool
    reason: str
    baselines: dict[str, LatencyBaseline] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceMatch:
    """How a device entry in the baseline file is recognized."""

    name_contains: str
    compute_capability: tuple[int, int]

    def matches(self, device_name: str, capability: tuple[int, int]) -> bool:
        return (self.name_contains.lower() in device_name.lower()
                and tuple(capability) == self.compute_capability)


@dataclass(frozen=True)
class LatencyPolicyTable:
    """Per-device latency policies, loaded from a JSON config file."""

    entries: tuple[tuple[DeviceMatch, DeviceLatencyPolicy], ...]

    @classmethod
    def load(cls, path: Path) -> LatencyPolicyTable:
        record = json.loads(Path(path).read_text())
        if record.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {record.get('schema_version')} != {CONFIG_SCHEMA_VERSION}")
        entries = []
        for device in record["devices"]:
            match = DeviceMatch(name_contains=str(device["match"]["name_contains"]),
                                compute_capability=tuple(int(x) for x in device["match"]["compute_capability"]))
            baselines = {precision: LatencyBaseline(p50_ms=float(b["p50_ms"]), margin=float(b["margin"]),
                                                    source=str(b["source"]))
                         for precision, b in device.get("baselines", {}).items()}
            entries.append((match, DeviceLatencyPolicy(device=str(device["device"]), gated=bool(device["gated"]),
                                                       reason=str(device["reason"]), baselines=baselines)))
        return cls(entries=tuple(entries))

    def resolve(self, device_name: str, capability: tuple[int, int]) -> DeviceLatencyPolicy:
        for match, policy in self.entries:
            if match.matches(device_name, capability):
                return policy
        return DeviceLatencyPolicy(
            device=f"unknown ({device_name}, sm_{capability[0]}{capability[1]})", gated=False,
            reason="no latency policy for this device in the baseline file")


class LatencyGate:
    """Judges a ``LatencySummary`` against one device policy."""

    def __init__(self, policy: DeviceLatencyPolicy) -> None:
        self.policy = policy

    def evaluate(self, precision: str, summary: LatencySummary) -> GateCheck:
        if not self.policy.gated:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{self.policy.device}: {self.policy.reason}")
        baseline = self.policy.baselines.get(precision)
        if baseline is None:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{self.policy.device}: no baseline for precision {precision!r}")
        passed = summary.p50_ms < baseline.limit_ms
        return GateCheck(LATENCY_CHECK, CHECK_PASS if passed else CHECK_FAIL, summary.p50_ms,
                         baseline.limit_ms,
                         f"p50 < {baseline.p50_ms} ms x (1 + {baseline.margin}); baseline: {baseline.source}")


@dataclass
class GateReport:
    """One gate run: checks, context and verdict."""

    precision: str
    device: str
    verdict: str
    reason: str
    checks: list[GateCheck]
    context: dict[str, object]
    latency: str
    latency_reason: str

    @classmethod
    def evaluated(cls, precision: str, device: str, checks: list[GateCheck],
                  context: dict[str, object], *, require_latency: bool = False) -> GateReport:
        """Verdict from the checks.

        ``latency`` is the status of the ``latency_p50`` check (``pass``,
        ``fail``, ``ungated``), or ``not_measured`` when there is none.
        Anything other than ``pass``/``fail`` is ungated: it is named in
        the reason, and with ``require_latency`` it makes the verdict
        ``blocked`` unless a check already failed.
        """
        failed = [c.name for c in checks if c.status == CHECK_FAIL]
        latency_check = next((c for c in checks if c.name == LATENCY_CHECK), None)
        latency = latency_check.status if latency_check is not None else LATENCY_NOT_MEASURED
        latency_reason = (latency_check.detail if latency_check is not None
                          else "no latency check in this run")
        ungated = latency not in (CHECK_PASS, CHECK_FAIL)
        ungated_note = f"latency {latency}: {latency_reason}"
        if failed:
            verdict = VERDICT_FAIL
            reason = f"failed: {', '.join(failed)}" + (f"; {ungated_note}" if ungated else "")
        elif ungated and require_latency:
            verdict = VERDICT_BLOCKED
            reason = f"latency required but {ungated_note}"
        else:
            verdict = VERDICT_PASS
            reason = "all gated checks passed" + (f"; {ungated_note}" if ungated else "")
        return cls(precision=precision, device=device, verdict=verdict, reason=reason,
                   checks=list(checks), context=dict(context), latency=latency,
                   latency_reason=latency_reason)

    @classmethod
    def not_run(cls, precision: str, device: str, verdict: str, reason: str,
                context: dict[str, object]) -> GateReport:
        if verdict not in (VERDICT_SKIPPED, VERDICT_BLOCKED):
            raise ValueError(f"not_run verdict must be skipped or blocked, got {verdict!r}")
        return cls(precision=precision, device=device, verdict=verdict, reason=reason,
                   checks=[], context=dict(context), latency=LATENCY_NOT_MEASURED,
                   latency_reason=f"not run: {verdict}")

    @property
    def exit_code(self) -> int:
        return 0 if self.verdict in (VERDICT_PASS, VERDICT_SKIPPED) else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "precision": self.precision,
            "device": self.device,
            "verdict": self.verdict,
            "reason": self.reason,
            "passed": self.verdict == VERDICT_PASS,
            "latency": self.latency,
            "latency_reason": self.latency_reason,
            "checks": [asdict(c) for c in self.checks],
            "context": self.context,
        }


def _check(name: str, passed: bool, value: float | bool, limit: float | bool, detail: str) -> GateCheck:
    return GateCheck(name, CHECK_PASS if passed else CHECK_FAIL, value, limit, detail)


def _at_least(name: str, value: float, limit: float) -> GateCheck:
    return _check(name, value >= limit, value, limit, f">= {limit}")

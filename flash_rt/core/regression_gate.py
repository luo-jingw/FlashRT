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
  policy. A baseline describes one configuration: the policy file
  (``LATENCY_SCHEMA_VERSION`` 2) holds, per device and per precision,
  NAMED configuration entries, each stating the configuration it
  describes (``config``) beside its ``p50_ms`` / ``margin`` / ``source``,
  and a run is compared with the entry of its own configuration name only.
  Latency is ungated when the device is marked ungated (for example a
  shared GPU whose timings are contaminated by another tenant), when the
  device matches no policy, when the run's configuration has no name,
  when a gated device has no baseline for the precision or for the
  configuration, when the entry is not seeded yet (``p50_ms`` null), or
  when the run's resolved configuration is not the one the entry states.
  An ungated latency is never silent: the report's top-level ``latency``
  field says ``ungated``, ``latency_reason`` says why (naming the
  configuration) and the verdict reason repeats it. For a missing or
  unseeded entry the report also carries ``latency_seed``: the measured
  P50 and the JSON entry to paste into the policy file. With
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
# The latency policy file is per configuration since version 2; version 1
# (one bound per precision) is refused, with no compatibility path.
LATENCY_SCHEMA_VERSION = 2
# The margin a freshly seeded entry gets when the file has none to reuse.
DEFAULT_SEED_MARGIN = 0.05
# An entry's `config` value that matches whatever the machine resolves
# (`use_fa4`: FA4 where it can run, the cuBLAS chain elsewhere).
CONFIG_AUTO = "auto"
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
    """One named configuration's recorded P50 and the allowed relative regression.

    ``config`` states the configuration the number describes (the resolved
    switches, in the terms the run reports them in); a value of ``"auto"``
    matches whatever the machine resolves. ``p50_ms`` is ``None`` while the
    entry is unseeded: it then gates nothing.
    """

    p50_ms: float | None
    margin: float
    source: str
    config: Mapping[str, object] = field(default_factory=dict)

    @property
    def seeded(self) -> bool:
        return self.p50_ms is not None

    @property
    def limit_ms(self) -> float:
        if self.p50_ms is None:
            raise ValueError("an unseeded baseline (p50_ms null) has no limit")
        return self.p50_ms * (1.0 + self.margin)


@dataclass(frozen=True)
class DeviceLatencyPolicy:
    """Whether latency is gated on one device, and its baselines.

    ``baselines[precision][configuration_name]`` is one entry.
    """

    device: str
    gated: bool
    reason: str
    baselines: dict[str, dict[str, LatencyBaseline]] = field(default_factory=dict)


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
        version = record.get("schema_version")
        if version != LATENCY_SCHEMA_VERSION:
            hint = (" (one bound per precision): latency baselines are per configuration now, each "
                    "precision holds named configuration entries with their `config`, p50_ms, margin "
                    "and source; re-seed the file in that form" if version == 1 else "")
            raise ValueError(f"{path}: latency baseline schema_version {version} is not supported, "
                             f"expected {LATENCY_SCHEMA_VERSION}{hint}")
        entries = []
        for device in record["devices"]:
            match = DeviceMatch(name_contains=str(device["match"]["name_contains"]),
                                compute_capability=tuple(int(x) for x in device["match"]["compute_capability"]))
            baselines = {
                precision: {name: _load_baseline(path, device["device"], precision, name, entry)
                            for name, entry in configurations.items()}
                for precision, configurations in device.get("baselines", {}).items()}
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

    def evaluate(self, precision: str, summary: LatencySummary, configuration: str | None,
                 resolved_config: Mapping[str, object] | None = None, *,
                 configuration_reason: str = "") -> GateCheck:
        """The ``latency_p50`` check of a run of ``configuration``.

        ``configuration`` is the run's baseline name, or ``None`` when the
        run's configuration is none of the named ones (``configuration_reason``
        says which configuration it is). ``resolved_config`` is what the run
        resolved, compared with the entry's ``config`` so an entry only
        judges a run of the configuration it describes.
        """
        device = self.policy.device
        if not self.policy.gated:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{device}: {self.policy.reason}")
        measured = f"measured P50 {summary.p50_ms:.2f} ms"
        if configuration is None:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{device}: this run's configuration has no latency baseline "
                             f"({configuration_reason or 'no configuration name'}); {measured}")
        configurations = self.policy.baselines.get(precision)
        if not configurations:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{device}: no baseline for precision {precision!r} "
                             f"(configuration {configuration!r}); {measured}")
        baseline = configurations.get(configuration)
        if baseline is None:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{device}: no baseline for configuration {configuration!r} at precision "
                             f"{precision!r} (known: {sorted(configurations)}); {measured}")
        if resolved_config is not None:
            differences = config_differences(baseline.config, resolved_config)
            if differences:
                return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                                 f"{device}: the baseline for configuration {configuration!r} at precision "
                                 f"{precision!r} describes {dict(baseline.config)}, this run resolved "
                                 f"{dict(resolved_config)} (differs in {differences}); {measured}")
        if not baseline.seeded:
            return GateCheck(LATENCY_CHECK, CHECK_UNGATED, summary.p50_ms, None,
                             f"{device}: baseline for configuration {configuration!r} at precision "
                             f"{precision!r} is not seeded yet (p50_ms null); {measured}; the report's "
                             f"latency_seed is the entry to paste")
        passed = summary.p50_ms < baseline.limit_ms
        return GateCheck(LATENCY_CHECK, CHECK_PASS if passed else CHECK_FAIL, summary.p50_ms,
                         baseline.limit_ms,
                         f"configuration {configuration!r}: p50 < {baseline.p50_ms} ms x "
                         f"(1 + {baseline.margin}); baseline: {baseline.source}")

    def seed_entry(self, precision: str, configuration: str | None, summary: LatencySummary,
                   config: Mapping[str, object], source: str,
                   resolved_config: Mapping[str, object] | None = None) -> dict[str, object] | None:
        """What to paste into the policy file to seed this run's baseline.

        ``None`` unless the latency is ungated for want of an entry: a gated
        device, a named configuration and no entry for it at ``precision``,
        or an entry that is not seeded yet and describes this run's
        configuration. Otherwise the record has the measured ``p50_ms`` and
        the JSON ``entry`` (``config``, ``p50_ms``, ``margin``, ``source``)
        to put at ``path`` in the file. An unseeded entry keeps its own
        ``config`` and margin; a new one takes ``config`` (the run's
        resolved configuration) and ``DEFAULT_SEED_MARGIN``.
        """
        if not self.policy.gated or configuration is None:
            return None
        existing = self.policy.baselines.get(precision, {}).get(configuration)
        if existing is not None:
            if existing.seeded:
                return None
            if resolved_config is not None and config_differences(existing.config, resolved_config):
                return None
        entry = {"config": dict(existing.config if existing is not None else config),
                 "p50_ms": round(summary.p50_ms, 2),
                 "margin": existing.margin if existing is not None else DEFAULT_SEED_MARGIN,
                 "source": source}
        return {"device": self.policy.device, "precision": precision, "configuration": configuration,
                "p50_ms": entry["p50_ms"],
                "path": f'devices["{self.policy.device}"].baselines["{precision}"]["{configuration}"]',
                "entry": entry}


def config_differences(described: Mapping[str, object], resolved: Mapping[str, object]) -> list[str]:
    """Keys of ``described`` (an entry's ``config``) the resolved configuration does not satisfy.

    A described value of ``"auto"`` matches any resolved value; every
    other value must be equal (``True`` is not ``1``); a key the resolved
    configuration lacks differs.
    """
    return sorted(key for key, want in described.items()
                  if want != CONFIG_AUTO and (key not in resolved or type(resolved[key]) is not type(want)
                                              or resolved[key] != want))


def _load_baseline(path: Path, device: str, precision: str, name: str,
                   entry: Mapping[str, object]) -> LatencyBaseline:
    where = f"{path}: {device}/{precision}/{name}"
    config = entry.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ValueError(f"{where}: `config` must state the configuration the baseline describes")
    p50 = entry.get("p50_ms")
    if p50 is not None and (isinstance(p50, bool) or not isinstance(p50, (int, float)) or p50 <= 0):
        raise ValueError(f"{where}: p50_ms must be a positive number or null (unseeded), got {p50!r}")
    return LatencyBaseline(p50_ms=None if p50 is None else float(p50), margin=float(entry["margin"]),
                           source=str(entry["source"]), config=dict(config))


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
    latency_seed: dict[str, object] | None = None

    @classmethod
    def evaluated(cls, precision: str, device: str, checks: list[GateCheck],
                  context: dict[str, object], *, require_latency: bool = False,
                  latency_seed: dict[str, object] | None = None) -> GateReport:
        """Verdict from the checks.

        ``latency`` is the status of the ``latency_p50`` check (``pass``,
        ``fail``, ``ungated``), or ``not_measured`` when there is none.
        Anything other than ``pass``/``fail`` is ungated: it is named in
        the reason, and with ``require_latency`` it makes the verdict
        ``blocked`` unless a check already failed. ``latency_seed`` is
        ``LatencyGate.seed_entry``'s record, carried into the report.
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
                   latency_reason=latency_reason, latency_seed=latency_seed)

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
            "latency_seed": self.latency_seed,
            "checks": [asdict(c) for c in self.checks],
            "context": self.context,
        }


def _check(name: str, passed: bool, value: float | bool, limit: float | bool, detail: str) -> GateCheck:
    return GateCheck(name, CHECK_PASS if passed else CHECK_FAIL, value, limit, detail)


def _at_least(name: str, value: float, limit: float) -> GateCheck:
    return _check(name, value >= limit, value, limit, f">= {limit}")

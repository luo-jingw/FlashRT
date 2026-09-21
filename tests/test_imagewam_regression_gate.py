"""CPU tests for the ImageWAM regression gate: policy, config files, fixture IO.

The gate runner itself (``tests/gate_imagewam_libero.py``) needs a GPU
and a generated fixture; everything it decides with is tested here on
synthetic inputs and on the committed config files. Its ``text_trim``
check is the exception: it is decided from the manifest alone, so the
committed v1 manifest exercises it.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from flash_rt.core.regression_gate import (
    CHECK_FAIL,
    CHECK_PASS,
    CHECK_UNGATED,
    DEFAULT_SEED_MARGIN,
    LATENCY_CHECK,
    LATENCY_NOT_MEASURED,
    LATENCY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    VERDICT_BLOCKED,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_SKIPPED,
    CosineSummary,
    DeviceLatencyPolicy,
    FidelityGate,
    FidelityMeasurement,
    FidelityThresholds,
    FidelityThresholdTable,
    GateCheck,
    GateReport,
    LatencyBaseline,
    LatencyGate,
    LatencyPolicyTable,
    LatencySummary,
    config_differences,
)
from flash_rt.datasets.imagewam_gate_fixture import (
    FIXTURE_FILE,
    FixtureManifest,
    GateFixtureStore,
    ImageWAMGateFixture,
)

CONFIG_DIR = Path(__file__).resolve().parent / "fixtures" / "imagewam_gate"
GATE_RUNNER = Path(__file__).resolve().parent / "gate_imagewam_libero.py"
BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
GENERATOR = BENCHMARKS / "imagewam_gate_fixture_generate.py"
COMPARE = BENCHMARKS / "imagewam_e2e_official_compare.py"
V1_MANIFEST = CONFIG_DIR / "imagewam_libero_gate_v1.manifest.json"
V2_MANIFEST = CONFIG_DIR / "imagewam_libero_gate_v2.manifest.json"


def gate_runner() -> types.ModuleType:
    """The gate runner as a module (it is a script, not a package module)."""
    name = "imagewam_gate_runner_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, GATE_RUNNER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


THRESHOLDS = FidelityThresholds(
    vs_official_median_min=0.997, vs_official_min_min=0.993,
    vs_fp16_reference_median_min=0.999, vs_fp16_reference_min_min=0.995,
    mae_vs_gt_ratio_max=1.02, requires_calibration=False, note="")


def _measurement(**overrides: object) -> FidelityMeasurement:
    values = dict(
        vs_official=CosineSummary(median=0.9984, minimum=0.9957, mean=0.998, count=40),
        vs_fp16_reference=CosineSummary(median=1.0, minimum=0.9999, mean=1.0, count=40),
        mae_vs_gt_mean=0.1836, reference_mae_vs_gt_mean=0.1836, all_finite=True)
    values.update(overrides)
    return FidelityMeasurement(**values)


def _by_name(checks: list[GateCheck]) -> dict[str, GateCheck]:
    return {c.name: c for c in checks}


# ── policy ──────────────────────────────────────────────────────────────


def test_cosine_summary():
    summary = CosineSummary.from_values([0.99, 0.999, 0.995])
    assert (summary.median, summary.minimum, summary.count) == (0.995, 0.99, 3)
    with pytest.raises(ValueError):
        CosineSummary.from_values([])


def test_fidelity_gate_passes_the_h100_fp16_baseline():
    checks = FidelityGate(THRESHOLDS).evaluate(_measurement())
    assert [c.status for c in checks] == [CHECK_PASS] * 6


@pytest.mark.parametrize("override, failing", [
    ({"vs_official": CosineSummary(0.996, 0.995, 0.996, 40)}, "vs_official_median"),
    ({"vs_official": CosineSummary(0.998, 0.99, 0.998, 40)}, "vs_official_min"),
    ({"vs_fp16_reference": CosineSummary(0.9985, 0.998, 0.998, 40)}, "vs_fp16_reference_median"),
    ({"vs_fp16_reference": CosineSummary(0.9999, 0.99, 0.999, 40)}, "vs_fp16_reference_min"),
    ({"mae_vs_gt_mean": 0.1836 * 1.03}, "mae_vs_gt_mean"),
    ({"all_finite": False}, "outputs_finite"),
])
def test_fidelity_gate_fails_each_bound(override, failing):
    checks = _by_name(FidelityGate(THRESHOLDS).evaluate(_measurement(**override)))
    assert checks[failing].status == CHECK_FAIL
    assert [name for name, c in checks.items() if c.status == CHECK_FAIL] == [failing]


def test_latency_summary_keeps_run_order():
    samples = [float(v) for group in range(10) for v in (group, group + 2)]
    summary = LatencySummary.from_samples(samples)
    assert summary.group_medians_ms == tuple(float(g + 1) for g in range(10))
    assert summary.iters == 20 and summary.min_ms == 0.0 and summary.max_ms == 11.0
    with pytest.raises(ValueError, match="at least 10"):
        LatencySummary.from_samples([1.0] * 9)


SERVED_CONFIG = {"text_trim": True, "use_fa4": "auto", "use_fa4_mot": "auto", "vae": "native_graph"}
UNTRIMMED_CONFIG = {"text_trim": False, "use_fa4": False, "use_fa4_mot": False, "vae": "torch"}


def _summary(p50: float) -> LatencySummary:
    return LatencySummary.from_samples([p50] * 10)


def _thor_policy(**named: LatencyBaseline) -> DeviceLatencyPolicy:
    return DeviceLatencyPolicy("thor", True, "target", {"nvfp4": dict(named)})


def test_latency_gate_rule_is_strict_less_than_limit():
    policy = _thor_policy(served_default=LatencyBaseline(200.0, 0.05, "test", SERVED_CONFIG))
    gate = LatencyGate(policy)
    assert gate.evaluate("nvfp4", _summary(209.9), "served_default").status == CHECK_PASS
    at_limit = gate.evaluate("nvfp4", _summary(210.0), "served_default")
    assert at_limit.status == CHECK_FAIL and at_limit.limit == pytest.approx(210.0)
    missing = gate.evaluate("fp16", _summary(1.0), "served_default")
    assert missing.status == CHECK_UNGATED and "no baseline" in missing.detail


def test_seeded_entry_passes_and_fails_against_its_own_bound():
    policy = _thor_policy(served_default=LatencyBaseline(120.0, 0.05, "test", SERVED_CONFIG),
                          untrimmed_reference=LatencyBaseline(200.0, 0.05, "test", UNTRIMMED_CONFIG))
    gate = LatencyGate(policy)
    served_config = dict(SERVED_CONFIG, use_fa4=True, use_fa4_mot=True)
    ok = gate.evaluate("nvfp4", _summary(125.0), "served_default", served_config)
    assert ok.status == CHECK_PASS and ok.limit == pytest.approx(126.0)
    assert "served_default" in ok.detail
    slow = gate.evaluate("nvfp4", _summary(127.0), "served_default", served_config)
    assert slow.status == CHECK_FAIL and slow.value == 127.0
    # the same 127 ms is nowhere near the untrimmed reference's bound: each
    # configuration is judged against its own number only
    reference = gate.evaluate("nvfp4", _summary(127.0), "untrimmed_reference", UNTRIMMED_CONFIG)
    assert reference.status == CHECK_PASS and reference.limit == pytest.approx(210.0)
    # a regression that the old single 202.2 ms bound would have let through on the served default
    assert gate.evaluate("nvfp4", _summary(180.0), "served_default", served_config).status == CHECK_FAIL


def test_unseeded_entry_is_ungated_not_passed_and_not_failed():
    policy = _thor_policy(served_default=LatencyBaseline(None, 0.05, "run the gate on Thor", SERVED_CONFIG))
    check = LatencyGate(policy).evaluate("nvfp4", _summary(125.86), "served_default",
                                         dict(SERVED_CONFIG, use_fa4=True, use_fa4_mot=True))
    assert check.status == CHECK_UNGATED and check.value == 125.86 and check.limit is None
    assert "served_default" in check.detail and "not seeded" in check.detail
    assert "125.86" in check.detail and "latency_seed" in check.detail
    with pytest.raises(ValueError, match="unseeded"):
        policy.baselines["nvfp4"]["served_default"].limit_ms
    report = GateReport.evaluated("nvfp4", "thor", [_fidelity_pass(), check], {})
    assert report.verdict == VERDICT_PASS and report.latency == CHECK_UNGATED
    assert "served_default" in report.latency_reason and "served_default" in report.reason
    required = GateReport.evaluated("nvfp4", "thor", [_fidelity_pass(), check], {}, require_latency=True)
    assert required.verdict == VERDICT_BLOCKED and "served_default" in required.reason


@pytest.mark.parametrize("configuration, reason_part", [
    ("fast_experiment", "no baseline for configuration 'fast_experiment'"),
    (None, "no latency baseline"),
])
def test_unknown_or_unnamed_configuration_is_ungated(configuration, reason_part):
    policy = _thor_policy(untrimmed_reference=LatencyBaseline(200.0, 0.05, "test", UNTRIMMED_CONFIG))
    check = LatencyGate(policy).evaluate("nvfp4", _summary(90.0), configuration, UNTRIMMED_CONFIG,
                                         configuration_reason="profile 'fast' resolved to ...")
    assert check.status == CHECK_UNGATED and check.value == 90.0
    assert reason_part in check.detail and "90.00" in check.detail
    if configuration is None:
        assert "profile 'fast' resolved to" in check.detail
    else:
        assert "untrimmed_reference" in check.detail   # the known names are listed
    required = GateReport.evaluated("nvfp4", "thor", [_fidelity_pass(), check], {}, require_latency=True)
    assert required.verdict == VERDICT_BLOCKED and reason_part in required.reason


def test_entry_only_judges_a_run_of_the_configuration_it_states():
    policy = _thor_policy(untrimmed_reference=LatencyBaseline(200.0, 0.05, "test", UNTRIMMED_CONFIG))
    gate = LatencyGate(policy)
    drifted = dict(UNTRIMMED_CONFIG, use_fa4=True)
    check = gate.evaluate("nvfp4", _summary(100.0), "untrimmed_reference", drifted)
    assert check.status == CHECK_UNGATED and "use_fa4" in check.detail and "untrimmed_reference" in check.detail
    assert gate.seed_entry("nvfp4", "untrimmed_reference", _summary(100.0), drifted, "src", drifted) is None


def test_config_differences_auto_matches_anything_and_bool_is_not_int():
    assert config_differences(SERVED_CONFIG, dict(SERVED_CONFIG, use_fa4=False, use_fa4_mot=True)) == []
    assert config_differences(SERVED_CONFIG, dict(SERVED_CONFIG, vae="torch")) == ["vae"]
    assert config_differences(UNTRIMMED_CONFIG, dict(UNTRIMMED_CONFIG, use_fa4=0)) == ["use_fa4"]
    assert config_differences(UNTRIMMED_CONFIG, {"text_trim": False}) == ["use_fa4", "use_fa4_mot", "vae"]


def test_latency_seed_carries_the_measured_p50_and_a_pasteable_entry():
    policy = _thor_policy(served_default=LatencyBaseline(None, 0.07, "seed me", SERVED_CONFIG))
    gate = LatencyGate(policy)
    run_config = dict(SERVED_CONFIG, use_fa4=True, use_fa4_mot=True)
    seed = gate.seed_entry("nvfp4", "served_default", _summary(118.4567), run_config, "gate run X", run_config)
    assert seed["p50_ms"] == 118.46 and seed["configuration"] == "served_default"
    assert seed["path"] == 'devices["thor"].baselines["nvfp4"]["served_default"]'
    # an unseeded entry keeps its own description and margin; the run only fills p50_ms and source
    assert seed["entry"] == {"config": SERVED_CONFIG, "p50_ms": 118.46, "margin": 0.07, "source": "gate run X"}
    record = json.loads(json.dumps(seed))
    assert record["entry"]["p50_ms"] == 118.46
    # pasting it in gives a seeded entry that gates: the round trip through the file format
    pasted = LatencyBaseline(**{**record["entry"]})
    assert pasted.seeded and pasted.limit_ms == pytest.approx(118.46 * 1.07)
    check = gate.evaluate("nvfp4", _summary(118.4567), "served_default", run_config)
    report = GateReport.evaluated("nvfp4", "thor", [_fidelity_pass(), check], {}, latency_seed=seed)
    assert report.to_dict()["latency_seed"]["entry"]["p50_ms"] == 118.46
    assert GateReport.evaluated("nvfp4", "thor", [_fidelity_pass()], {}).to_dict()["latency_seed"] is None


def test_latency_seed_for_a_missing_entry_uses_the_runs_config_and_default_margin():
    gate = LatencyGate(_thor_policy(untrimmed_reference=LatencyBaseline(200.0, 0.05, "test", UNTRIMMED_CONFIG)))
    config = dict(SERVED_CONFIG, use_fa4=True, use_fa4_mot=False)
    seed = gate.seed_entry("nvfp4", "served_default", _summary(120.0), config, "src", config)
    assert seed["entry"] == {"config": config, "p50_ms": 120.0, "margin": DEFAULT_SEED_MARGIN, "source": "src"}
    assert LatencyGate(DeviceLatencyPolicy("thor", True, "t")).seed_entry(
        "fp16", "served_default", _summary(1.0), config, "src", config)["path"].endswith(
        '["fp16"]["served_default"]')


@pytest.mark.parametrize("policy, configuration", [
    # seeded entry: the number is already there
    (_thor_policy(served_default=LatencyBaseline(120.0, 0.05, "t", SERVED_CONFIG)), "served_default"),
    # no configuration name: nothing to seed
    (_thor_policy(), None),
    # an ungated device is not seeded from
    (DeviceLatencyPolicy("h100", False, "shared GPU"), "served_default"),
])
def test_no_latency_seed_when_there_is_nothing_to_seed(policy, configuration):
    assert LatencyGate(policy).seed_entry("nvfp4", configuration, _summary(1.0), SERVED_CONFIG, "s",
                                          SERVED_CONFIG) is None


def test_latency_gate_ungated_device_still_records_p50():
    check = LatencyGate(DeviceLatencyPolicy("h100", False, "shared GPU")).evaluate(
        "fp16", LatencySummary.from_samples([500.0] * 10), "served_default")
    assert check.status == CHECK_UNGATED and check.value == 500.0 and "shared GPU" in check.detail


def test_report_verdicts_and_exit_codes():
    passing = GateReport.evaluated("fp16", "h100", [GateCheck("a", CHECK_PASS, 1.0, 1.0, ""),
                                                   GateCheck(LATENCY_CHECK, CHECK_PASS, 2.0, 3.0, "")], {})
    failing = GateReport.evaluated("fp16", "h100", [GateCheck("a", CHECK_FAIL, 0.0, 1.0, "")], {})
    skipped = GateReport.not_run("fp8_static", "h100", VERDICT_SKIPPED, "no calibration file", {})
    blocked = GateReport.not_run("fp8_static", "h100", VERDICT_BLOCKED, "missing keyword", {})
    assert [(r.verdict, r.exit_code) for r in (passing, failing, skipped, blocked)] == [
        (VERDICT_PASS, 0), (VERDICT_FAIL, 1), (VERDICT_SKIPPED, 0), (VERDICT_BLOCKED, 1)]
    record = json.loads(json.dumps(failing.to_dict()))
    assert record["schema_version"] == RESULT_SCHEMA_VERSION
    assert record["passed"] is False
    assert record["reason"] == "failed: a; latency not_measured: no latency check in this run"
    with pytest.raises(ValueError):
        GateReport.not_run("fp16", "h100", VERDICT_PASS, "", {})
    assert passing.to_dict()["latency"] == CHECK_PASS and passing.reason == "all gated checks passed"
    assert skipped.to_dict()["latency"] == LATENCY_NOT_MEASURED


def _fidelity_pass() -> GateCheck:
    return GateCheck("vs_official_median", CHECK_PASS, 0.999, 0.997, "")


@pytest.mark.parametrize("policy, precision, reason_part", [
    (DeviceLatencyPolicy("h100", False, "shared GPU"), "fp16", "shared GPU"),
    (_thor_policy(served_default=LatencyBaseline(231.6, 0.05, "OPT-015", SERVED_CONFIG)), "fp16",
     "no baseline for precision"),
    (LatencyPolicyTable(entries=()).resolve("NVIDIA GeForce RTX 4090", (8, 9)), "nvfp4",
     "no latency policy for this device"),
])
def test_ungated_latency_is_explicit_and_can_be_required(policy, precision, reason_part):
    latency = LatencyGate(policy).evaluate(precision, LatencySummary.from_samples([100.0] * 10),
                                           "served_default")
    default = GateReport.evaluated(precision, policy.device, [_fidelity_pass(), latency], {})
    record = default.to_dict()
    assert default.verdict == VERDICT_PASS and default.exit_code == 0
    assert record["latency"] == CHECK_UNGATED and reason_part in record["latency_reason"]
    assert "latency ungated" in record["reason"] and reason_part in record["reason"]
    required = GateReport.evaluated(precision, policy.device, [_fidelity_pass(), latency], {},
                                    require_latency=True)
    assert required.verdict == VERDICT_BLOCKED and required.exit_code == 1
    assert reason_part in required.reason


def test_required_latency_does_not_mask_a_fidelity_failure():
    latency = LatencyGate(DeviceLatencyPolicy("h100", False, "shared GPU")).evaluate(
        "fp16", LatencySummary.from_samples([100.0] * 10), "served_default")
    report = GateReport.evaluated("fp16", "h100", [GateCheck("vs_official_min", CHECK_FAIL, 0.9, 0.99, ""),
                                                   latency], {}, require_latency=True)
    assert report.verdict == VERDICT_FAIL and "latency ungated" in report.reason


def test_missing_latency_check_counts_as_ungated():
    report = GateReport.evaluated("fp16", "h100", [_fidelity_pass()], {}, require_latency=True)
    assert report.latency == LATENCY_NOT_MEASURED and report.verdict == VERDICT_BLOCKED


# ── committed config files ──────────────────────────────────────────────


def test_committed_latency_baselines():
    table = LatencyPolicyTable.load(CONFIG_DIR / "latency_baselines.json")
    thor = table.resolve("NVIDIA Thor", (11, 0))
    assert thor.gated and set(thor.baselines) == {"nvfp4"}
    named = thor.baselines["nvfp4"]
    assert set(named) == {"served_default", "untrimmed_reference"}
    # The 202.2 ms record from the eccf14f round (gate 202.2 ms, three consecutive
    # end-to-end repeats within 0.4 ms, issues.md ISSUE-082) describes the
    # untrimmed, FA4-off configuration only.
    untrimmed = named["untrimmed_reference"]
    assert untrimmed.p50_ms == 202.2 and untrimmed.margin == 0.05
    assert untrimmed.limit_ms == pytest.approx(202.2 * 1.05)
    assert untrimmed.config == {"text_trim": False, "use_fa4": False, "use_fa4_mot": False, "vae": "torch"}
    assert "UNTRIMMED" in untrimmed.source and "eccf14f" in untrimmed.source
    assert "both configurations" not in untrimmed.source
    # The served default is seeded from its own Thor gate run (0921, three repeats
    # 105.42 / 105.93 / 105.66 ms): a run of that configuration is judged against
    # 105.42 ms, never against the untrimmed number.
    served = named["served_default"]
    assert served.seeded and served.p50_ms == 105.42 and served.margin == 0.05
    assert served.limit_ms == pytest.approx(105.42 * 1.05)
    assert served.config == {"text_trim": True, "use_fa4": "auto", "use_fa4_mot": "auto",
                             "vae": "native_graph"}
    assert "de0ef510050f" in served.source and "fa4_fallback_reason=None" in served.source
    run = dict(served.config, use_fa4=True, use_fa4_mot=True)
    assert LatencyGate(thor).evaluate("nvfp4", _summary(105.9), "served_default", run).status == CHECK_PASS
    slow = LatencyGate(thor).evaluate("nvfp4", _summary(125.86), "served_default", run)
    assert slow.status == CHECK_FAIL and slow.value == 125.86   # the old default's number now fails
    h100 = table.resolve("NVIDIA H100 NVL", (9, 0))
    assert h100.device == "h100" and not h100.gated and h100.baselines == {}
    unknown = table.resolve("NVIDIA GeForce RTX 4090", (8, 9))
    assert not unknown.gated and unknown.device.startswith("unknown")


def test_committed_baseline_configs_are_named_by_the_gate_from_the_same_terms():
    """Each committed entry's ``config`` is a configuration the runner names as that entry."""
    runner = gate_runner()
    table = LatencyPolicyTable.load(CONFIG_DIR / "latency_baselines.json")
    named = table.resolve("NVIDIA Thor", (11, 0)).baselines["nvfp4"]
    assert dict(named["untrimmed_reference"].config) == runner.UNTRIMMED_REFERENCE_CONFIG
    assert runner.baseline_configuration("native", {"use_fa4": False}, dict(named["untrimmed_reference"].config))[0] \
        == "untrimmed_reference"
    assert runner.baseline_configuration("default", {}, dict(named["served_default"].config))[0] == "served_default"


def test_latency_baselines_of_schema_version_1_are_refused(tmp_path):
    old = tmp_path / "latency_baselines.json"
    old.write_text(json.dumps({
        "schema_version": 1, "rule": "p50 < baseline * (1 + margin)",
        "devices": [{"device": "thor", "match": {"name_contains": "Thor", "compute_capability": [11, 0]},
                     "gated": True, "reason": "target",
                     "baselines": {"nvfp4": {"p50_ms": 202.2, "margin": 0.05, "source": "old"}}}]}))
    with pytest.raises(ValueError) as refused:
        LatencyPolicyTable.load(old)
    message = str(refused.value)
    assert "schema_version 1" in message and f"expected {LATENCY_SCHEMA_VERSION}" in message
    assert "per configuration" in message and str(old) in message


def _baseline_file(tmp_path, entry):
    path = tmp_path / "latency_baselines.json"
    path.write_text(json.dumps({
        "schema_version": LATENCY_SCHEMA_VERSION, "rule": "r",
        "devices": [{"device": "thor", "match": {"name_contains": "Thor", "compute_capability": [11, 0]},
                     "gated": True, "reason": "target", "baselines": {"nvfp4": {"served_default": entry}}}]}))
    return path


@pytest.mark.parametrize("entry, message", [
    ({"p50_ms": 1.0, "margin": 0.05, "source": "s"}, "`config` must state"),
    ({"config": {}, "p50_ms": 1.0, "margin": 0.05, "source": "s"}, "`config` must state"),
    ({"config": {"text_trim": True}, "p50_ms": 0, "margin": 0.05, "source": "s"}, "positive number or null"),
    ({"config": {"text_trim": True}, "p50_ms": "fast", "margin": 0.05, "source": "s"}, "positive number or null"),
])
def test_malformed_latency_baseline_entries_are_refused(tmp_path, entry, message):
    with pytest.raises(ValueError, match=message):
        LatencyPolicyTable.load(_baseline_file(tmp_path, entry))


def test_latency_baseline_entry_round_trips_seeded_and_unseeded(tmp_path):
    seeded = LatencyPolicyTable.load(_baseline_file(
        tmp_path, {"config": {"text_trim": True}, "p50_ms": 120, "margin": 0.1, "source": "s"}))
    entry = seeded.resolve("Thor", (11, 0)).baselines["nvfp4"]["served_default"]
    assert entry.p50_ms == 120.0 and entry.limit_ms == pytest.approx(132.0) and entry.seeded
    unseeded = LatencyPolicyTable.load(_baseline_file(
        tmp_path, {"config": {"text_trim": True}, "p50_ms": None, "margin": 0.1, "source": "s"}))
    assert not unseeded.resolve("Thor", (11, 0)).baselines["nvfp4"]["served_default"].seeded


def test_committed_fidelity_thresholds():
    table = FidelityThresholdTable.load(CONFIG_DIR / "fidelity_thresholds.json")
    assert set(table.precisions) == {"fp16", "nvfp4", "fp8_static", "fp8_static_cutlass",
                                     "e0m3_hadamard"}
    assert table.for_precision("fp8_static").requires_calibration is True
    assert table.for_precision("nvfp4").requires_calibration is False
    assert table.for_precision("fp8") is None
    # every served precision the gate can be asked for has an entry: the gate
    # looks a precision up by name and reports "no fidelity thresholds" otherwise
    cutlass = table.for_precision("fp8_static_cutlass")
    static = table.for_precision("fp8_static")
    assert cutlass is not None and cutlass.requires_calibration is True
    assert (cutlass.vs_official_median_min, cutlass.vs_official_min_min) == (
        static.vs_official_median_min, static.vs_official_min_min)
    e0m3 = table.for_precision("e0m3_hadamard")
    assert e0m3 is not None and e0m3.requires_calibration is False
    nvfp4 = table.for_precision("nvfp4")
    assert (e0m3.vs_official_median_min, e0m3.vs_official_min_min) == (
        nvfp4.vs_official_median_min, nvfp4.vs_official_min_min)
    fp16 = table.for_precision("fp16")
    # The documented H100 fp16 end-to-end baseline must pass its own gate.
    checks = FidelityGate(fp16).evaluate(_measurement())
    assert all(c.status == CHECK_PASS for c in checks)


def test_committed_fixture_manifests_are_well_formed():
    manifests = sorted(CONFIG_DIR.glob("*.manifest.json"))
    for path in manifests:
        manifest = FixtureManifest.read(path)
        assert set(manifest.arrays) == set(ImageWAMGateFixture.array_names()), path
        assert FIXTURE_FILE in manifest.files
        assert manifest.name == path.name.removesuffix(".manifest.json")


# ── the fixture generator's imports ─────────────────────────────────────


def _module_level_names(path: Path) -> set[str]:
    """Names a module binds at its top level, read from the source (no import)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def test_generator_imports_names_that_exist():
    """Every name the generator imports resolves, without running it.

    The generator needs a GPU, a LIBERO dataset and the end-to-end script's
    environment, so nothing here imports it: its names come from the
    compare script (checked against that script's own source) and from
    ``flash_rt`` (checked by importing the modules, which is CPU-safe).
    """
    tree = ast.parse(GENERATOR.read_text())
    compare_names = _module_level_names(COMPARE)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == "imagewam_e2e_official_compare":
            missing = [a.name for a in node.names if a.name not in compare_names]
        elif node.module.startswith("flash_rt."):
            module = importlib.import_module(node.module)
            missing = [a.name for a in node.names if not hasattr(module, a.name)]
        else:
            continue
        assert not missing, f"{node.module}: {missing}"


# ── the fixture's text_trim against the gate's configuration ─────────────


def test_committed_v1_manifest_predates_the_text_trim_field():
    record = json.loads(V1_MANIFEST.read_text())
    assert "text_trim" not in record
    assert "text_trim" not in record["metadata"]["fp16_reference"]
    assert FixtureManifest.read(V1_MANIFEST).text_trim is False


def test_absent_text_trim_in_a_manifest_means_untrimmed():
    record = json.loads(V1_MANIFEST.read_text())
    assert FixtureManifest.from_json(json.dumps(record)).text_trim is False
    assert FixtureManifest.from_json(json.dumps(record | {"text_trim": True})).text_trim is True


def test_gate_accepts_the_committed_v1_manifest_untrimmed():
    manifest = FixtureManifest.read(V1_MANIFEST)
    assert gate_runner().text_trim_mismatch(manifest, False) is None


def test_gate_parser_defaults_are_the_served_configuration():
    """A bare gate run gates what is served: the trimmed fixture v2 and
    trimming on, so its defaults are self-consistent (fixture and switch
    agree). ``--no-text-trim`` selects the untrimmed reference, and
    ``--manifest`` stays overridable for it."""
    runner = gate_runner()
    parser = runner.build_parser()
    required = ["--precision", "nvfp4", "--fixture-dir", "/fixture"]
    served = parser.parse_args(required)
    assert served.text_trim is True
    assert served.manifest == V2_MANIFEST == runner.DEFAULT_MANIFEST
    assert parser.parse_args(required + ["--text-trim"]).text_trim is True
    manifest = FixtureManifest.read(served.manifest)
    assert manifest.name == "imagewam_libero_gate_v2" and manifest.text_trim is True
    assert runner.text_trim_mismatch(manifest, served.text_trim) is None
    reference = parser.parse_args(required + ["--no-text-trim"])
    assert reference.text_trim is False and reference.manifest == V2_MANIFEST
    overridden = parser.parse_args(required + ["--no-text-trim", "--manifest", str(V1_MANIFEST)])
    assert overridden.manifest == V1_MANIFEST and overridden.text_trim is False
    assert runner.text_trim_mismatch(FixtureManifest.read(overridden.manifest), overridden.text_trim) is None


def test_gate_refuses_a_text_trim_mismatch_naming_both_values():
    runner = gate_runner()
    untrimmed = FixtureManifest.read(V1_MANIFEST)
    trimmed = dataclasses.replace(untrimmed, text_trim=True)
    reason = runner.text_trim_mismatch(untrimmed, True)
    assert "imagewam_libero_gate_v1" in reason
    assert "text_trim=False" in reason and "text_trim=True" in reason
    other = runner.text_trim_mismatch(trimmed, False)
    assert "imagewam_libero_gate_v1" in other and "text_trim=True" in other and "text_trim=False" in other
    assert runner.text_trim_mismatch(trimmed, True) is None


def test_gate_runs_the_switch_it_checks():
    """The value compared with the fixture is the value handed to the frontend."""
    tree = ast.parse(GATE_RUNNER.read_text())
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "--text-trim" in strings
    constructor_calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "load_imagewam"]
    assert constructor_calls
    assert all("text_trim" in {kw.arg for kw in call.keywords} for call in constructor_calls)
    # and the frontend is not built around the deployment entry any more
    assert not [node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ImageWAMTorchFrontendThor"]


# ── the runner's configuration name, overrides and dims check ─────────────


@pytest.mark.parametrize("item, expected", [
    ("use_fa4=false", ("use_fa4", False)),
    ("use_fa4_mot=TRUE", ("use_fa4_mot", True)),
    ("use_fa4=auto", ("use_fa4", None)),
    ("vae_graph=auto", ("vae_graph", None)),
    ("vae_encoder=auto", ("vae_encoder", "auto")),      # a real value of that key, not the None spelling
    ("vae_encoder=torch", ("vae_encoder", "torch")),
    ("text_trim_cache_size=16", ("text_trim_cache_size", 16)),
    ("awq_scope=adaln+down", ("awq_scope", "adaln+down")),
    ("awq_alpha=0.5", ("awq_alpha", "0.5")),            # parsed literally: not an int, so a string
])
def test_runner_parses_overrides_literally(item, expected):
    assert gate_runner().parse_override(item) == expected


@pytest.mark.parametrize("item, message", [
    ("use_fa4", "KEY=VALUE"),
    ("=true", "KEY=VALUE"),
    ("use_fa4=", "KEY=VALUE"),
    ("use_fa5=true", "not an expert key"),
    ("text_trim=false", "--no-text-trim"),
])
def test_runner_refuses_malformed_overrides(item, message):
    with pytest.raises(ValueError, match=message):
        gate_runner().parse_override(item)


def test_runner_collects_overrides_and_refuses_a_repeated_key():
    runner = gate_runner()
    assert runner.parse_overrides(["use_fa4=false", "vae_graph=false"]) == {"use_fa4": False, "vae_graph": False}
    assert runner.parse_overrides([]) == {}
    with pytest.raises(ValueError, match="more than once"):
        runner.parse_overrides(["use_fa4=false", "use_fa4=true"])


def test_gate_parser_profile_and_override_flags():
    parser = gate_runner().build_parser()
    required = ["--precision", "nvfp4", "--fixture-dir", "/fixture"]
    bare = parser.parse_args(required)
    assert bare.profile == "default" and bare.override == []
    both = parser.parse_args(required + ["--profile", "native", "--override", "use_fa4=false",
                                         "--override", "vae_graph=false"])
    assert both.profile == "native" and both.override == ["use_fa4=false", "vae_graph=false"]


def _libero_options(profile="default", ae_model_path="/ae", **expert):
    from flash_rt.models.imagewam.config_resolver import resolve_config
    from flash_rt.models.imagewam.structure import ImageWAMStructure
    from flash_rt.models.imagewam.workload import ImageWAMWorkload
    return resolve_config(ImageWAMWorkload.libero(), ImageWAMStructure.libero(), profile=profile,
                          ae_model_path=ae_model_path, **expert).options


def test_runner_names_the_served_default_from_the_resolved_profile():
    """The profile the gate builds by default resolves to the configuration the
    committed ``served_default`` entry states, and the runner names it so."""
    runner = gate_runner()
    table = LatencyPolicyTable.load(CONFIG_DIR / "latency_baselines.json")
    entry = table.resolve("NVIDIA Thor", (11, 0)).baselines["nvfp4"]["served_default"]
    options = _libero_options()
    config = runner.resolved_configuration(options)
    assert config == {"text_trim": True, "use_fa4": "auto", "use_fa4_mot": "auto", "vae": "native_graph"}
    assert config_differences(entry.config, config) == []
    name, _ = runner.baseline_configuration("default", {}, config)
    assert name == "served_default"
    # the runtime-resolved FA4 values replace the unresolved ones, as in the effective_config line
    ran = runner.resolved_configuration(options, use_fa4=True, use_fa4_mot=False)
    assert ran["use_fa4"] is True and ran["use_fa4_mot"] is False
    assert config_differences(entry.config, ran) == []
    assert runner.baseline_configuration("default", {}, ran)[0] == "served_default"


def test_runner_names_the_untrimmed_reference_from_the_resolved_options():
    runner = gate_runner()
    expert = {"use_fa4": False, "use_fa4_mot": False, "vae_graph": False, "vae_encoder": "torch"}
    config = runner.resolved_configuration(_libero_options(text_trim=False, **expert))
    assert config == runner.UNTRIMMED_REFERENCE_CONFIG
    assert runner.baseline_configuration("default", expert, config)[0] == "untrimmed_reference"
    # `native` states FA4 off and the torch VAE itself, so text_trim off is all it needs
    native = runner.resolved_configuration(_libero_options("native", text_trim=False))
    assert runner.baseline_configuration("native", {}, native)[0] == "untrimmed_reference"


@pytest.mark.parametrize("profile, expert, overrides, reason_part", [
    # untrimmed but FA4/VAE left to the machine: not the FA4-off reference
    ("default", {"text_trim": False}, {}, "neither 'served_default'"),
    # the served default with an override is a different configuration
    ("default", {"use_fa4": False}, {"use_fa4": False}, "neither 'served_default'"),
    ("default", {"vae_graph": False}, {"vae_graph": False}, "neither 'served_default'"),
    # another profile that trims
    ("fast", {}, {}, "neither 'served_default'"),
    ("native", {}, {}, "neither 'served_default'"),
    # an override outside the four switches changes the computation
    ("default", {"gemm_variant_autotune": True}, {"gemm_variant_autotune": True}, "beyond the switches"),
    ("native", {"text_trim": False, "gemm_variant_autotune": True}, {"gemm_variant_autotune": True},
     "beyond the switches"),
])
def test_runner_gives_no_baseline_name_to_any_other_configuration(profile, expert, overrides, reason_part):
    runner = gate_runner()
    config = runner.resolved_configuration(_libero_options(profile, **expert))
    name, reason = runner.baseline_configuration(profile, overrides, config)
    assert name is None
    assert reason_part in reason
    if reason_part == "neither 'served_default'":
        assert str(config) in reason   # the reason names the resolved configuration


def test_runner_unnamed_configuration_is_ungated_naming_the_resolved_configuration():
    runner = gate_runner()
    table = LatencyPolicyTable.load(CONFIG_DIR / "latency_baselines.json")
    thor = table.resolve("NVIDIA Thor", (11, 0))
    config = runner.resolved_configuration(_libero_options("fast"), use_fa4=True, use_fa4_mot=True)
    name, reason = runner.baseline_configuration("fast", {}, config)
    check = LatencyGate(thor).evaluate("nvfp4", _summary(100.0), name, config, configuration_reason=reason)
    assert check.status == CHECK_UNGATED
    assert "'fast'" in check.detail and "native_graph" in check.detail
    assert LatencyGate(thor).seed_entry("nvfp4", name, _summary(100.0), config, "s", config) is None


@pytest.mark.parametrize("manifest_path", [V1_MANIFEST, V2_MANIFEST])
def test_committed_fixture_dims_are_the_libero_workloads_resolved_dims(manifest_path):
    from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
    dims = FixtureManifest.read(manifest_path).metadata["fp16_reference"]["dims"]
    runner = gate_runner()
    assert runner.dims_mismatch(dims, LIBERO_REAL_DIMS) == {}
    assert runner.dims_mismatch(dict(dims, hidden=dims["hidden"] + 1), LIBERO_REAL_DIMS) == {
        "hidden": (dims["hidden"] + 1, dims["hidden"])}
    assert runner.dims_mismatch({"not_a_dim": 1}, LIBERO_REAL_DIMS) == {"not_a_dim": (1, None)}


def test_runner_seed_source_states_what_the_run_was_measured_under():
    source = gate_runner().seed_source(
        stamp="20260921T101500Z", git={"commit": "0123456789abcdef", "clean": True},
        device_name="NVIDIA Thor", capability=(11, 0),
        clock_state={"nvpmodel_mode": "MAXN", "gpu_locked": False, "emc_locked": None},
        warmup=20, iters=100, fixture="imagewam_libero_gate_v2",
        effective_config="effective_config precision=nvfp4 text_trim=True")
    for part in ("20260921T101500Z", "0123456789ab", "NVIDIA Thor sm_110", "nvpmodel MAXN", "emc_locked=None",
                 "--iters 100", "imagewam_libero_gate_v2", "effective_config precision=nvfp4"):
        assert part in source
    assert "not clean" not in source
    assert "worktree not clean" in gate_runner().seed_source(
        stamp="s", git={"commit": "c", "clean": False}, device_name="d", capability=(1, 2),
        clock_state={}, warmup=0, iters=10, fixture="f", effective_config="e")


# ── fixture IO ──────────────────────────────────────────────────────────


def _tiny_fixture(n: int = 3, seeds: int = 2, horizon: int = 4, tasks: int = 2,
                  text_trim: bool = False) -> ImageWAMGateFixture:
    rng = np.random.default_rng(0)
    chunk = (n, seeds, horizon, 7)
    gt = rng.standard_normal((n, horizon, 7)).astype(np.float32)
    gt[-1, -1] = np.nan
    return ImageWAMGateFixture(
        text_trim=text_trim,
        view1=rng.integers(0, 256, (n, 8, 8, 3), dtype=np.uint8),
        view2=rng.integers(0, 256, (n, 8, 8, 3), dtype=np.uint8),
        state=rng.standard_normal((n, 8)).astype(np.float32),
        task_index=np.arange(n, dtype=np.int64) % tasks,
        episode=np.arange(n, dtype=np.int64), frame=np.zeros(n, dtype=np.int64),
        gt_actions=gt, gt_len=np.array([horizon] * (n - 1) + [horizon - 1], dtype=np.int64),
        prompts=np.array([f"task {i}" for i in range(tasks)]),
        context_bf16_bits=rng.integers(0, 2**16, (tasks, 5, 6), dtype=np.uint16),
        context_mask=np.ones((tasks, 5), dtype=bool),
        seeds=np.arange(seeds, dtype=np.int64),
        noise=rng.standard_normal(chunk).astype(np.float32),
        official_actions=rng.standard_normal(chunk).astype(np.float32),
        fp16_reference_actions=rng.standard_normal(chunk).astype(np.float32))


def test_fixture_array_names_are_every_field_but_text_trim():
    fields = tuple(f.name for f in dataclasses.fields(ImageWAMGateFixture) if f.name != "text_trim")
    assert ImageWAMGateFixture.array_names() == fields


def test_fixture_round_trip(tmp_path):
    fixture = _tiny_fixture()
    store = GateFixtureStore(tmp_path)
    manifest = store.save(fixture, "tiny_v1", {"suite": "synthetic"})
    reread = FixtureManifest.read(tmp_path / "manifest.json")
    assert reread == manifest
    loaded = store.load(reread)
    for name, array in fixture.arrays().items():
        np.testing.assert_array_equal(getattr(loaded, name), array)
    assert loaded.prompts.tolist() == ["task 0", "task 1"]


@pytest.mark.parametrize("text_trim", [False, True])
def test_text_trim_round_trips_through_the_manifest(tmp_path, text_trim):
    fixture = _tiny_fixture(text_trim=text_trim)
    store = GateFixtureStore(tmp_path)
    manifest = store.save(fixture, "tiny_v1", {"suite": "synthetic"})
    assert manifest.text_trim is text_trim
    assert json.loads((tmp_path / "manifest.json").read_text())["text_trim"] is text_trim
    assert store.load(FixtureManifest.read(tmp_path / "manifest.json")).text_trim is text_trim


def test_fixture_validation_rejects_a_non_bool_text_trim():
    fixture = _tiny_fixture()
    fixture.text_trim = 1
    with pytest.raises(ValueError, match="text_trim"):
        fixture.validate()


def test_fixture_file_tamper_is_rejected(tmp_path):
    store = GateFixtureStore(tmp_path)
    manifest = store.save(_tiny_fixture(), "tiny_v1", {})
    raw = bytearray(store.fixture_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    store.fixture_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="does not match manifest"):
        store.load(manifest)


def test_fixture_array_mismatch_is_rejected(tmp_path):
    store = GateFixtureStore(tmp_path)
    manifest = store.save(_tiny_fixture(), "tiny_v1", {})
    other = GateFixtureStore(tmp_path / "other").save(
        _tiny_fixture(n=3, seeds=2), "tiny_v1", {})
    doctored = FixtureManifest(
        name=manifest.name, format_version=manifest.format_version, files=manifest.files,
        arrays=dict(manifest.arrays, noise=other.arrays["official_actions"]), metadata={},
        text_trim=manifest.text_trim)
    with pytest.raises(ValueError, match="noise"):
        store.load(doctored)


def test_fixture_validation_catches_inconsistent_shapes():
    fixture = _tiny_fixture()
    fixture.noise = fixture.noise[:, :1]
    with pytest.raises(ValueError, match="noise"):
        fixture.validate()
    fixture = _tiny_fixture()
    fixture.task_index = fixture.task_index + 5
    with pytest.raises(ValueError, match="task_index"):
        fixture.validate()

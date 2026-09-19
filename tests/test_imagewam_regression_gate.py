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
    LATENCY_CHECK,
    LATENCY_NOT_MEASURED,
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


def test_latency_gate_rule_is_strict_less_than_limit():
    policy = DeviceLatencyPolicy("thor", True, "target",
                                 {"nvfp4": LatencyBaseline(p50_ms=200.0, margin=0.05, source="test")})
    gate = LatencyGate(policy)
    assert gate.evaluate("nvfp4", LatencySummary.from_samples([209.9] * 10)).status == CHECK_PASS
    at_limit = gate.evaluate("nvfp4", LatencySummary.from_samples([210.0] * 10))
    assert at_limit.status == CHECK_FAIL and at_limit.limit == pytest.approx(210.0)
    missing = gate.evaluate("fp16", LatencySummary.from_samples([1.0] * 10))
    assert missing.status == CHECK_UNGATED and "no baseline" in missing.detail


def test_latency_gate_ungated_device_still_records_p50():
    check = LatencyGate(DeviceLatencyPolicy("h100", False, "shared GPU")).evaluate(
        "fp16", LatencySummary.from_samples([500.0] * 10))
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
    (DeviceLatencyPolicy("thor", True, "target",
                         {"nvfp4": LatencyBaseline(231.6, 0.05, "OPT-015")}), "fp16", "no baseline for precision"),
    (LatencyPolicyTable(entries=()).resolve("NVIDIA GeForce RTX 4090", (8, 9)), "nvfp4",
     "no latency policy for this device"),
])
def test_ungated_latency_is_explicit_and_can_be_required(policy, precision, reason_part):
    latency = LatencyGate(policy).evaluate(precision, LatencySummary.from_samples([100.0] * 10))
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
        "fp16", LatencySummary.from_samples([100.0] * 10))
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
    assert thor.gated and thor.baselines["nvfp4"].p50_ms == 231.6
    assert thor.baselines["nvfp4"].limit_ms == pytest.approx(231.6 * 1.05)
    h100 = table.resolve("NVIDIA H100 NVL", (9, 0))
    assert h100.device == "h100" and not h100.gated
    unknown = table.resolve("NVIDIA GeForce RTX 4090", (8, 9))
    assert not unknown.gated and unknown.device.startswith("unknown")


def test_committed_fidelity_thresholds():
    table = FidelityThresholdTable.load(CONFIG_DIR / "fidelity_thresholds.json")
    assert set(table.precisions) == {"fp16", "nvfp4", "fp8_static"}
    assert table.for_precision("fp8_static").requires_calibration is True
    assert table.for_precision("nvfp4").requires_calibration is False
    assert table.for_precision("fp8") is None
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
                         and node.func.id == "ImageWAMTorchFrontendThor"]
    assert constructor_calls
    assert all("text_trim" in {kw.arg for kw in call.keywords} for call in constructor_calls)


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

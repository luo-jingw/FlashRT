"""``flash_rt.hardware.jetson_clock_state`` against fake sysfs trees.

Each test builds a Jetson-like (or plain) filesystem under ``tmp_path``
and a fake ``nvpmodel -q`` runner, so the probe runs on any
machine. Node names follow Thor's devfreq layout (``gpu-gpc-0``,
``gpu-nvd-0``) as used by ``tests/bench_pi05_decoder_fp4_e2e.py``.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from flash_rt.hardware.jetson_clock_state import (
    JetsonClockProbe,
    ToolQuery,
    report_jetson_clock_state,
)

GPU_MAX_HZ = 1_575_000_000
EMC_HZ = 4_266_000_000
MAXN_QUERY = "NV Power Mode: MAXN\n0\n"


class FakeRunner:
    """Answers ``nvpmodel -q`` from a table and records every call."""

    def __init__(self, answers: dict[str, ToolQuery]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Sequence[str]) -> ToolQuery:
        self.calls.append(tuple(argv))
        return self.answers.get(argv[0], ToolQuery(False, f"{argv[0]}: not found on PATH"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _devfreq(root: Path, name: str, *, cur: int, low: int, high: int,
             governor: str = "nvhost_podgov") -> None:
    node = root / "sys/class/devfreq" / name
    _write(node / "cur_freq", f"{cur}\n")
    _write(node / "min_freq", f"{low}\n")
    _write(node / "max_freq", f"{high}\n")
    _write(node / "governor", f"{governor}\n")


def _thor_tree(root: Path, *, gpu_cur: int = GPU_MAX_HZ, gpu_min: int = GPU_MAX_HZ,
               with_emc_node: bool = False, emc_cur: int = EMC_HZ) -> Path:
    _write(root / "etc/nv_tegra_release", "# R38 (release), REVISION: 2.0\n")
    _write(root / "proc/device-tree/model", "NVIDIA Jetson AGX Thor Developer Kit\x00")
    _write(root / "proc/device-tree/compatible", "nvidia,p3971-0080+p3834-0008\x00nvidia,tegra264\x00")
    for name in ("gpu-gpc-0", "gpu-nvd-0"):
        _devfreq(root, name, cur=gpu_cur, low=gpu_min, high=GPU_MAX_HZ)
    _devfreq(root, "15340000.vic", cur=1, low=1, high=9)  # not a GPU/EMC node: ignored
    _write(root / "sys/kernel/nvpmodel_clk_cap/emc", f"{EMC_HZ}\n")
    _write(root / "sys/kernel/nvpmodel_clk_cap/gpu", f"{GPU_MAX_HZ}\n")
    if with_emc_node:
        _devfreq(root, "emc", cur=emc_cur, low=EMC_HZ, high=EMC_HZ, governor="performance")
    return root


def _runner(nvpmodel_text: str | None = MAXN_QUERY) -> FakeRunner:
    answers = {}
    if nvpmodel_text is not None:
        answers["nvpmodel"] = ToolQuery(True, nvpmodel_text)
    return FakeRunner(answers)


def test_not_a_jetson(tmp_path):
    runner = _runner()
    state = JetsonClockProbe(tmp_path, runner).read()
    assert state.is_jetson is False
    assert state.locked is False
    assert "not a Jetson" in state.platform
    assert state.gpu == () and state.warnings == ()
    assert runner.calls == []  # no tool is run off-Jetson


def test_pinned_thor(tmp_path):
    runner = _runner()
    state = JetsonClockProbe(_thor_tree(tmp_path), runner).read()
    print(json.dumps(state.to_dict(), indent=1))
    assert state.is_jetson and state.platform.startswith("NVIDIA Jetson AGX Thor")
    assert [n.name for n in state.gpu] == ["gpu-gpc-0", "gpu-nvd-0"]
    assert all(n.locked and n.cur_hz == GPU_MAX_HZ for n in state.gpu)
    assert state.gpu_locked is True
    assert state.nvpmodel_mode == "MAXN" and state.nvpmodel_mode_id == 0
    assert state.power_mode_max is True
    assert state.emc_locked is None  # no EMC devfreq node: unobservable, not unlocked
    assert state.locked is True
    assert dict(state.clock_caps_hz) == {"emc": EMC_HZ, "gpu": GPU_MAX_HZ}
    assert state.warnings == ()
    assert runner.calls == [("nvpmodel", "-q")]  # the only tool run; no sudo, no jetson_clocks


def test_dynamic_gpu_clocks_at_maxn_are_recorded_not_warned(tmp_path):
    """MAXN with DVFS-managed clocks is the expected serving state."""
    root = _thor_tree(tmp_path, gpu_cur=306_000_000, gpu_min=306_000_000)
    state = JetsonClockProbe(root, _runner()).read()
    assert state.power_mode_max is True
    assert state.gpu_locked is False and state.locked is False
    assert [n.cur_hz for n in state.gpu] == [306_000_000, 306_000_000]
    assert state.warnings == ()


def test_non_maxn_power_mode(tmp_path):
    state = JetsonClockProbe(_thor_tree(tmp_path), _runner("NV Power Mode: 50W\n2\n")).read()
    assert state.nvpmodel_mode == "50W" and state.nvpmodel_mode_id == 2
    assert state.power_mode_max is False
    assert state.gpu_locked is True and state.locked is False
    assert any("not MAXN" in w for w in state.warnings)
    assert not any("sudo" in w for w in state.warnings)


def test_missing_nvpmodel_is_unobservable_not_unlocked(tmp_path):
    state = JetsonClockProbe(_thor_tree(tmp_path), _runner(None)).read()
    assert state.nvpmodel.available is False
    assert state.power_mode_max is None
    assert state.locked is True  # GPU pinned; power mode unknown is a warning only
    assert any("nvpmodel mode unobservable" in w for w in state.warnings)


def test_emc_devfreq_node_participates(tmp_path):
    locked = JetsonClockProbe(_thor_tree(tmp_path / "a", with_emc_node=True), _runner()).read()
    assert locked.emc_locked is True and locked.locked is True and locked.warnings == ()
    unlocked = JetsonClockProbe(
        _thor_tree(tmp_path / "b", with_emc_node=True, emc_cur=2_133_000_000), _runner()).read()
    assert unlocked.emc_locked is False and unlocked.locked is False
    assert unlocked.warnings == ()


def test_jetson_without_gpu_devfreq(tmp_path):
    root = tmp_path
    _write(root / "proc/device-tree/compatible", "nvidia,tegra234\x00")
    state = JetsonClockProbe(root, _runner()).read()
    assert state.is_jetson is True
    assert state.gpu_locked is False and state.locked is False
    assert any("no GPU devfreq node" in w for w in state.warnings)


def test_unreadable_frequency_is_not_locked(tmp_path):
    root = _thor_tree(tmp_path)
    (root / "sys/class/devfreq/gpu-nvd-0/cur_freq").write_text("garbage\n")
    state = JetsonClockProbe(root, _runner()).read()
    nvd = [n for n in state.gpu if n.name == "gpu-nvd-0"][0]
    assert nvd.cur_hz is None and nvd.locked is False and state.locked is False


def test_report_prints_record_and_summary(tmp_path):
    lines: list[str] = []
    root = _thor_tree(tmp_path, gpu_cur=306_000_000, gpu_min=306_000_000)
    state = report_jetson_clock_state(JetsonClockProbe(root, _runner()), emit=lines.append)
    record = json.loads(lines[0].removeprefix("[jetson-clock-state] "))
    assert record["locked"] is False and record["is_jetson"] is True
    assert record["gpu"][0]["name"] == "gpu-gpc-0" and record["gpu"][0]["locked"] is False
    assert record["clock_caps_hz"] == {"emc": EMC_HZ, "gpu": GPU_MAX_HZ}
    assert "power mode MAXN" in lines[1] and "dynamic" in lines[1]
    assert len(lines) == 2 + len(state.warnings)
    assert not any("sudo" in line or "jetson_clocks" in line for line in lines)


def test_report_off_jetson(tmp_path):
    lines: list[str] = []
    state = report_jetson_clock_state(JetsonClockProbe(tmp_path, _runner()), emit=lines.append)
    assert state.is_jetson is False
    assert json.loads(lines[0].removeprefix("[jetson-clock-state] "))["is_jetson"] is False
    assert "not a Jetson" in lines[1]


def test_this_machine_reads_without_error():
    """Observational: the real root on whatever machine runs the suite."""
    state = JetsonClockProbe().read()
    print(json.dumps(state.to_dict(), sort_keys=True))
    json.dumps(state.to_dict())

"""Jetson power-mode and clock-lock state, read before a latency benchmark.

A Jetson GPU's frequency is set by devfreq (DVFS). Unless the power mode
is MAXN and ``jetson_clocks`` has pinned ``min_freq == max_freq``, the
GPU and EMC clocks move with load, and a latency number depends on the
clock state it happened to see. This module reads that state and returns
it as a structured record so a benchmark can print it next to its
numbers.

Sources, all read-only:

* Jetson detection: ``/etc/nv_tegra_release`` or a ``nvidia,tegra``
  entry in ``/proc/device-tree/compatible``.
* ``nvpmodel -q``: power-mode name (``NV Power Mode: MAXN``) and id.
* ``jetson_clocks --show``: the tool's own report. It needs root on most
  L4T releases; without root the record says so. The tool is never run
  through ``sudo``.
* ``/sys/class/devfreq/<name>/{cur_freq,min_freq,max_freq,governor}``:
  GPU nodes (names containing ``gpu``, e.g. Thor's ``gpu-gpc-0`` and
  ``gpu-nvd-0``, or a Tegra GPU id such as ``17000000.ga10b``) and EMC
  nodes (names containing ``emc``).
* ``/sys/kernel/nvpmodel_clk_cap/*``: the clock caps the power mode sets
  (Thor exposes ``emc`` here).

Lock verdict (``JetsonClockState.locked``): the machine is a Jetson, at
least one GPU devfreq node is visible and every GPU node has
``cur == min == max``, no visible EMC devfreq node is unlocked, and the
nvpmodel mode, when readable, is MAXN. Anything unlocked or unobservable
adds a warning.

Pi0.5's end-to-end benchmark (``tests/bench_pi05_decoder_fp4_e2e.py``,
``machine_state``) applies the same GPU rule and raises instead of
warning. This module only reports; the caller decides.

The module is stdlib-only, so it imports on any machine.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

_TOOL_TIMEOUT_S = 10.0
_TOOL_TEXT_LIMIT = 4000
_GPU_NAME_MARKERS = ("gpu", "gp10b", "gv11b", "ga10b", "gb10b")
_EMC_NAME_MARKER = "emc"
_RECORD_PREFIX = "[jetson-clock-state]"


@dataclass(frozen=True)
class ToolQuery:
    """One external tool invocation: whether it ran, and its text.

    ``available`` is ``False`` when the binary is missing, timed out, or
    exited non-zero; ``text`` then says why.
    """

    available: bool
    text: str


@dataclass(frozen=True)
class DevfreqNode:
    """One ``/sys/class/devfreq/<name>`` entry. Frequencies in Hz."""

    name: str
    kind: str
    cur_hz: int | None
    min_hz: int | None
    max_hz: int | None
    governor: str | None

    @property
    def locked(self) -> bool:
        values = (self.cur_hz, self.min_hz, self.max_hz)
        return None not in values and len(set(values)) == 1


@dataclass(frozen=True)
class JetsonClockState:
    """Power and clock record for one benchmark run."""

    is_jetson: bool
    platform: str
    nvpmodel: ToolQuery
    nvpmodel_mode: str | None
    nvpmodel_mode_id: int | None
    jetson_clocks: ToolQuery
    gpu: tuple[DevfreqNode, ...]
    emc: tuple[DevfreqNode, ...]
    clock_caps_hz: tuple[tuple[str, int], ...]
    gpu_locked: bool
    emc_locked: bool | None
    power_mode_max: bool | None
    locked: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form, with each devfreq node's ``locked``."""
        record = asdict(self)
        record["gpu"] = [dict(asdict(n), locked=n.locked) for n in self.gpu]
        record["emc"] = [dict(asdict(n), locked=n.locked) for n in self.emc]
        record["clock_caps_hz"] = dict(self.clock_caps_hz)
        record["warnings"] = list(self.warnings)
        return record


class CommandRunner(Protocol):
    """Runs one read-only tool invocation."""

    def run(self, argv: Sequence[str]) -> ToolQuery:
        ...


class SubprocessCommandRunner:
    """Runs a tool with a timeout, without a shell and without ``sudo``."""

    def run(self, argv: Sequence[str]) -> ToolQuery:
        if shutil.which(argv[0]) is None:
            return ToolQuery(False, f"{argv[0]}: not found on PATH")
        try:
            done = subprocess.run(
                list(argv), check=False, capture_output=True, text=True,
                timeout=_TOOL_TIMEOUT_S, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolQuery(False, f"{' '.join(argv)}: {type(exc).__name__}: {exc}")
        text = (done.stdout + done.stderr).strip()[:_TOOL_TEXT_LIMIT]
        if done.returncode != 0:
            return ToolQuery(False, f"{' '.join(argv)}: exit {done.returncode}: {text}")
        return ToolQuery(True, text)


class JetsonClockProbe:
    """Reads the clock state under a filesystem root.

    ``root`` is ``/`` on a real machine; tests pass a temporary directory
    holding a fake sysfs tree. ``runner`` executes ``nvpmodel`` and
    ``jetson_clocks``.
    """

    def __init__(self, root: Path = Path("/"), runner: CommandRunner | None = None) -> None:
        self._root = Path(root)
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def read(self) -> JetsonClockState:
        platform = self._platform()
        if not self._is_jetson():
            reason = "not a Jetson: no /etc/nv_tegra_release and no nvidia,tegra device-tree entry"
            return JetsonClockState(
                is_jetson=False, platform=platform or reason,
                nvpmodel=ToolQuery(False, "not queried: not a Jetson"),
                nvpmodel_mode=None, nvpmodel_mode_id=None,
                jetson_clocks=ToolQuery(False, "not queried: not a Jetson"),
                gpu=(), emc=(), clock_caps_hz=(),
                gpu_locked=False, emc_locked=None, power_mode_max=None,
                locked=False, warnings=())

        nvpmodel = self._runner.run(("nvpmodel", "-q"))
        mode, mode_id = self._parse_nvpmodel(nvpmodel)
        jetson_clocks = self._runner.run(("jetson_clocks", "--show"))
        gpu, emc = self._devfreq_nodes()
        caps = self._clock_caps()

        gpu_locked = bool(gpu) and all(n.locked for n in gpu)
        emc_locked = None if not emc else all(n.locked for n in emc)
        power_mode_max = None if mode is None else mode.upper().startswith("MAXN")
        locked = gpu_locked and emc_locked is not False and power_mode_max is not False
        warnings = self._warnings(nvpmodel, mode, power_mode_max, gpu, emc)
        return JetsonClockState(
            is_jetson=True, platform=platform or "Jetson (model unreadable)",
            nvpmodel=nvpmodel, nvpmodel_mode=mode, nvpmodel_mode_id=mode_id,
            jetson_clocks=jetson_clocks, gpu=gpu, emc=emc, clock_caps_hz=caps,
            gpu_locked=gpu_locked, emc_locked=emc_locked,
            power_mode_max=power_mode_max, locked=locked, warnings=warnings)

    def _path(self, absolute: str) -> Path:
        return self._root / absolute.lstrip("/")

    def _read_text(self, absolute: str | Path) -> str | None:
        path = absolute if isinstance(absolute, Path) else self._path(absolute)
        try:
            return path.read_bytes().decode("utf-8", errors="replace").replace("\x00", " ").strip()
        except OSError:
            return None

    def _read_int(self, path: Path) -> int | None:
        text = self._read_text(path)
        if text is None:
            return None
        try:
            return int(text.split()[0])
        except (ValueError, IndexError):
            return None

    def _platform(self) -> str:
        return self._read_text("/proc/device-tree/model") or ""

    def _is_jetson(self) -> bool:
        if self._path("/etc/nv_tegra_release").is_file():
            return True
        compatible = self._read_text("/proc/device-tree/compatible") or ""
        return "nvidia,tegra" in compatible

    @staticmethod
    def _parse_nvpmodel(query: ToolQuery) -> tuple[str | None, int | None]:
        if not query.available:
            return None, None
        lines = [line.strip() for line in query.text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if line.startswith("NV Power Mode:"):
                mode = line.split(":", 1)[1].strip()
                mode_id = None
                if index + 1 < len(lines) and lines[index + 1].isdigit():
                    mode_id = int(lines[index + 1])
                return mode, mode_id
        return None, None

    def _devfreq_nodes(self) -> tuple[tuple[DevfreqNode, ...], tuple[DevfreqNode, ...]]:
        base = self._path("/sys/class/devfreq")
        gpu: list[DevfreqNode] = []
        emc: list[DevfreqNode] = []
        if not base.is_dir():
            return (), ()
        for entry in sorted(base.iterdir()):
            name = entry.name
            lower = name.lower()
            if _EMC_NAME_MARKER in lower:
                kind, bucket = "emc", emc
            elif any(marker in lower for marker in _GPU_NAME_MARKERS):
                kind, bucket = "gpu", gpu
            else:
                continue
            bucket.append(DevfreqNode(
                name=name, kind=kind,
                cur_hz=self._read_int(entry / "cur_freq"),
                min_hz=self._read_int(entry / "min_freq"),
                max_hz=self._read_int(entry / "max_freq"),
                governor=self._read_text(entry / "governor")))
        return tuple(gpu), tuple(emc)

    def _clock_caps(self) -> tuple[tuple[str, int], ...]:
        base = self._path("/sys/kernel/nvpmodel_clk_cap")
        if not base.is_dir():
            return ()
        caps = []
        for entry in sorted(base.iterdir()):
            value = self._read_int(entry) if entry.is_file() else None
            if value is not None:
                caps.append((entry.name, value))
        return tuple(caps)

    @staticmethod
    def _warnings(nvpmodel: ToolQuery, mode: str | None, power_mode_max: bool | None,
                  gpu: tuple[DevfreqNode, ...], emc: tuple[DevfreqNode, ...]) -> tuple[str, ...]:
        warnings = []
        if mode is None:
            warnings.append(f"nvpmodel mode unobservable ({nvpmodel.text[:200]})")
        elif not power_mode_max:
            warnings.append(f"nvpmodel mode is {mode!r}, not MAXN (sudo nvpmodel -m 0)")
        if not gpu:
            warnings.append("no GPU devfreq node under /sys/class/devfreq; GPU clock lock unobservable")
        for node in gpu + emc:
            if not node.locked:
                warnings.append(
                    f"{node.kind} devfreq {node.name} not locked: cur={node.cur_hz} "
                    f"min={node.min_hz} max={node.max_hz} governor={node.governor} (sudo jetson_clocks)")
        if not emc:
            warnings.append("no EMC devfreq node under /sys/class/devfreq; EMC clock lock unobservable")
        return tuple(warnings)


def report_jetson_clock_state(probe: JetsonClockProbe | None = None,
                              emit: Callable[[str], None] = print) -> JetsonClockState:
    """Read the clock state, print it, and return it.

    Prints one ``[jetson-clock-state] <json>`` line, then one ``WARNING``
    line per warning when the clocks are not locked, or one line saying
    the machine is not a Jetson.
    """
    state = (probe if probe is not None else JetsonClockProbe()).read()
    emit(f"{_RECORD_PREFIX} {json.dumps(state.to_dict(), sort_keys=True)}")
    if not state.is_jetson:
        emit(f"{_RECORD_PREFIX} not a Jetson ({os.uname().nodename}); "
             f"clock locking does not apply and latency here is not a Jetson number")
        return state
    if state.locked:
        emit(f"{_RECORD_PREFIX} clocks locked ({state.nvpmodel_mode})")
    else:
        emit(f"{_RECORD_PREFIX} WARNING: clocks NOT locked; latency below is not a locked-clock number")
    for warning in state.warnings:
        emit(f"{_RECORD_PREFIX} WARNING: {warning}")
    return state

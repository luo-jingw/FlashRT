"""ImageWAM fidelity + latency regression gate on a versioned LIBERO fixture.

For one precision, builds the frontend through the deployment entry
(``flash_rt.frontends.torch.imagewam_thor.load_imagewam`` on
``ImageWAMWorkload.libero()``, ``--profile``, default ``default``), runs its
served ``infer()`` (real checkpoint, real VAE, proprio, dataset stats) on
every fixture observation and seed, starting from the fixture's fixed initial
noise, and checks:

* fidelity against the fixture's official ImageWAM reference and its
  FlashRT ``fp16`` reference (per-sample cosine median/min in normalized
  action space; mean MAE against ground truth in real units, bounded by
  the fp16 reference's own MAE), thresholds from
  ``tests/fixtures/imagewam_gate/fidelity_thresholds.json``;
* latency of the served ``infer()`` (default noise) against the device
  policy in ``tests/fixtures/imagewam_gate/latency_baselines.json``
  (``p50 < baseline * (1 + margin)``; ungated on the shared H100), per
  configuration (see "Latency baselines" below).

Fidelity is measured with the fixture's fixed N(0,1) initial noise, the
noise the official sampler draws for each seed, passed through
``infer(obs, action_noise=...)``. It is not the served default draw,
``0.01 * N(0,1)`` (issues.md ISSUE-002). With the served draw, fp16 on
fixture v1 measures vs-official median 0.99683 and min 0.98591 over the
40 runs, below this gate's fp16 bounds (0.997 / 0.993); with the fixed
N(0,1) noise it measures 0.99836 / 0.99554. The latency loop does use
the served default draw.

The latency check records the Jetson clock state and never refuses
dynamic clocks: Thor runs as is, at MAXN with DVFS-managed clocks, and
the latency baseline is measured in that state (issues.md ISSUE-061).

The official model and Qwen3 are not loaded: the fixture carries the
official Qwen3 context. Gate policy: ``flash_rt/core/regression_gate.py``;
fixture format: ``flash_rt/datasets/imagewam_gate_fixture.py``; fixture
generation: ``benchmarks/imagewam_gate_fixture_generate.py``.

The fixture's ``text_trim`` must match the configuration under test
(``--text-trim`` / ``--no-text-trim``, default on): the ``fp16`` reference
was recorded with that switch, and comparing a run against a reference
recorded the other way measures the switch, not the precision (issues.md
ISSUE-080). A mismatch is ``blocked``, naming the fixture, the fixture's
value and the one under test, before the checkpoint is hashed.

The gate's own defaults are the served configuration: fixture v2
(``imagewam_libero_gate_v2``, the trimmed fp16 reference) and trimming on, so
a bare run gates what is served. The untrimmed reference stays reachable and
unchanged: fixture v1 (``imagewam_libero_gate_v1``) with ``--no-text-trim``
and v1's manifest (``--manifest`` defaults to v2's, and stays overridable).

The configuration under test is the profile's, resolved by ``load_imagewam``:
profile ``default`` is the fastest configuration the machine and the inputs
allow (text_trim, FA4 at both attention sites where the machine can run it,
the cuBLAS chain elsewhere, and the native VAE encoder inside the CUDA graph
because the autoencoder path is given). ``--override KEY=VALUE`` (repeatable)
sets an expert key (``config_resolver.EXPERT_KEYS``): ``true``/``false`` are
bools, integers are ints, anything else is a string, and ``use_fa4``,
``use_fa4_mot`` and ``vae_graph`` also take ``auto``. ``--text-trim`` /
``--no-text-trim`` is the ``text_trim`` override (not repeated through
``--override``). The resolved ``effective_config`` line (the runtime-resolved
FA4 values and any capture-time fallback included) is printed and recorded in
``result.json`` under ``context["config"]["effective_config"]``.

Latency baselines are per configuration (``latency_baselines.json``, schema 2):
the run's configuration name is derived from what it resolved, never from a
flag. Profile ``default`` with no ``--override`` and text_trim on is
``served_default``; text_trim off with FA4 off at both sites and the torch VAE
outside the graph (for example ``--no-text-trim --override use_fa4=false
--override use_fa4_mot=false --override vae_graph=false --override
vae_encoder=torch``, or ``--profile native --no-text-trim``) is
``untrimmed_reference``; anything else has no baseline and its latency is
ungated, with the resolved configuration named in the reason. An entry that
is not seeded yet (``p50_ms`` null) is also ungated: the result then carries
``latency_seed``, the measured P50 and the JSON entry to paste into the
baselines file (printed on the console too). Seeding ``served_default`` is
one run of the bare command below on the Thor.

``fp8_static`` (thresholds marked ``requires_calibration``) is gated only
with a real activation-calibration file, given by ``--fp8-calibration``
or ``$IMAGEWAM_FP8_CALIBRATION``:

* no path, or the file does not exist: verdict ``skipped`` (exit 0);
* the file exists and ``ImageWAMTorchFrontendThor.__init__`` declares a
  ``calibration_path`` keyword (the calibration stream's name for it):
  the path is passed as ``load_imagewam(calibration_path=...)`` and the
  precision is gated like any other;
* the file exists but the constructor has no such keyword: verdict
  ``blocked`` (exit 1). The gate never runs ``fp8_static`` on the
  placeholder ``N(0, 0.1)`` calibration.

Latency is ungated on a device marked ungated (the shared H100), on a
device with no policy, for a precision or a configuration with no baseline
(or an unseeded one) on a gated device, and for a run whose configuration is
neither named configuration. The result then carries ``latency: "ungated"`` and
``latency_reason`` at top level and the verdict reason names it; the
verdict stays ``pass``. ``--require-latency`` turns an ungated latency
into ``blocked`` (exit 1).

Writes ``<output-dir>/result.json`` and prints one ``__IMAGEWAM_GATE__``
JSON line. Exit code 0 for ``pass``/``skipped``, 1 for ``fail``/``blocked``.

Inputs are checked before any GPU work: ``--iters`` must be at least
``LATENCY_GROUP_COUNT`` (10), and the checkpoint must match the
manifest's SHA-256 (about 9 GB hashed once per run). ``--skip-checkpoint-hash``
replaces the hash with a size-only check and is recorded in the result;
a manifest without a checkpoint hash requires that flag. A mismatch, or
a ``dataset_stats.json`` that differs from the fixture's, is ``blocked``.

Required env: ``CKPT_PATH`` (``dataset_stats.json`` beside it),
``FLUX2_AE_MODEL_PATH`` (or ``AE_MODEL_PATH``), ``FLUX2_SRC``. The served
default, fixture v2 and trimming on, with no flags::

    python tests/gate_imagewam_libero.py --precision nvfp4 \\
        --fixture-dir /path/to/imagewam_libero_gate_v2

and the untrimmed reference, the same run against fixture v1 with
``--no-text-trim``, v1's manifest and the FA4-off, torch-VAE switches::

    python tests/gate_imagewam_libero.py --precision nvfp4 --no-text-trim \\
        --override use_fa4=false --override use_fa4_mot=false \\
        --override vae_graph=false --override vae_encoder=torch \\
        --manifest tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json \\
        --fixture-dir /path/to/imagewam_libero_gate_v1
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from flash_rt.core.parity import parity_metrics  # noqa: E402
from flash_rt.core.regression_gate import (  # noqa: E402
    LATENCY_GROUP_COUNT,
    VERDICT_BLOCKED,
    VERDICT_SKIPPED,
    CosineSummary,
    FidelityGate,
    FidelityMeasurement,
    FidelityThresholdTable,
    GateReport,
    LatencyGate,
    LatencyPolicyTable,
    LatencySummary,
)
from flash_rt.datasets.imagewam_gate_fixture import (  # noqa: E402
    FIXTURE_FILE,
    FixtureManifest,
    GateFixtureStore,
    ImageWAMGateFixture,
)
from flash_rt.frontends.torch.imagewam_thor import (  # noqa: E402
    ImageWAMTorchFrontendThor,
    load_imagewam,
)
from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state  # noqa: E402
from flash_rt.models.imagewam.config_resolver import (  # noqa: E402
    EXPERT_KEYS,
    format_effective_config,
)
from flash_rt.models.imagewam.dataset_stats import MinMaxNormalizer, load_real_normalizers  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

CONFIG_DIR = REPO / "tests" / "fixtures" / "imagewam_gate"
# The served configuration's fixture: trimmed (v2). The untrimmed reference is
# imagewam_libero_gate_v1, reached with --no-text-trim and --manifest.
DEFAULT_MANIFEST = CONFIG_DIR / "imagewam_libero_gate_v2.manifest.json"
FIXTURE_DIR_ENV = "IMAGEWAM_GATE_FIXTURE_DIR"
FP8_CALIBRATION_ENV = "IMAGEWAM_FP8_CALIBRATION"
# The constructor keyword a real fp8_static calibration file is handed to.
FP8_CALIBRATION_FRONTEND_KWARG = "calibration_path"
DEV = "cuda"
BF16 = torch.bfloat16
RESULT_PREFIX = "__IMAGEWAM_GATE__ "

# The latency baselines' configuration names (latency_baselines.json).
SERVED_DEFAULT = "served_default"
UNTRIMMED_REFERENCE = "untrimmed_reference"
SERVED_PROFILE = "default"
# The expert keys that make up a baseline's `config` (regression_gate's
# entries state text_trim, use_fa4, use_fa4_mot and vae); an override of any
# other key changes the computation without being part of the description.
CONFIG_OVERRIDE_KEYS = ("text_trim", "use_fa4", "use_fa4_mot", "vae_encoder", "vae_graph")
# Keys whose value `auto` means "resolve at construction" (None in the resolver).
AUTO_OVERRIDE_KEYS = ("use_fa4", "use_fa4_mot", "vae_graph")
UNTRIMMED_REFERENCE_CONFIG = {"text_trim": False, "use_fa4": False, "use_fa4_mot": False, "vae": "torch"}


def parse_override(item: str) -> tuple[str, object]:
    """One ``--override KEY=VALUE`` as ``(key, value)``.

    The key must be an expert key (``config_resolver.EXPERT_KEYS``). The
    value is parsed literally: ``true`` / ``false`` (any case) are bools, an
    integer literal is an int, ``auto`` is ``None`` for ``use_fa4``,
    ``use_fa4_mot`` and ``vae_graph`` (resolve at construction), and
    anything else is a string. ``text_trim`` is refused: it has its own flag.
    """
    key, sep, text = item.partition("=")
    key, text = key.strip(), text.strip()
    if not sep or not key or not text:
        raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
    if key not in EXPERT_KEYS:
        raise ValueError(f"--override {key!r} is not an expert key; known: {', '.join(EXPERT_KEYS)}")
    if key == "text_trim":
        raise ValueError("text_trim is set with --text-trim / --no-text-trim, not --override")
    lowered = text.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    if lowered == "auto" and key in AUTO_OVERRIDE_KEYS:
        return key, None
    try:
        return key, int(text)
    except ValueError:
        return key, text


def parse_overrides(items: list[str]) -> dict[str, object]:
    """Every ``--override`` as one dict; a key given twice is an error."""
    overrides: dict[str, object] = {}
    for item in items:
        key, value = parse_override(item)
        if key in overrides:
            raise ValueError(f"--override {key!r} given more than once")
        overrides[key] = value
    return overrides


def resolved_configuration(options: object, *, use_fa4: bool | None = None,
                           use_fa4_mot: bool | None = None) -> dict[str, object]:
    """The four switches a latency baseline states, from the resolved options.

    The terms of the ``effective_config`` line: ``use_fa4`` / ``use_fa4_mot``
    are the runtime-resolved values when given (``fe.use_fa4``), else the
    options' own, ``auto`` while unresolved; ``vae`` is the encoder, with
    ``_graph`` appended when it runs inside the CUDA graph (``torch``,
    ``native``, ``native_graph``).
    """
    fa4 = options.use_fa4 if use_fa4 is None else use_fa4
    mot = options.use_fa4_mot if use_fa4_mot is None else use_fa4_mot
    graph = options.vae_graph_input is not None
    return {"text_trim": bool(options.text_trim),
            "use_fa4": "auto" if fa4 is None else fa4,
            "use_fa4_mot": "auto" if mot is None else mot,
            "vae": f"{options.vae_encoder}_graph" if graph else options.vae_encoder}


def baseline_configuration(profile: str, overrides: dict[str, object],
                           config: dict[str, object]) -> tuple[str | None, str]:
    """The latency baseline's configuration name for a run, and why.

    ``profile`` and ``overrides`` (the ``--override`` dict, without
    ``text_trim``) are what was asked for; ``config`` is what resolved
    (``resolved_configuration``). ``served_default``: profile ``default``,
    no override, text_trim on. ``untrimmed_reference``: text_trim off, FA4
    off at both sites, torch VAE, and no override outside those switches.
    Anything else has no name (``None``), and the reason names the resolved
    configuration.
    """
    other = sorted(set(overrides) - set(CONFIG_OVERRIDE_KEYS))
    if other:
        return None, (f"override(s) {other} change the computation beyond the switches a baseline "
                      f"describes; resolved {config}")
    if profile == SERVED_PROFILE and not overrides and config["text_trim"] is True:
        return SERVED_DEFAULT, f"profile {SERVED_PROFILE!r}, no override, text_trim on"
    if config == UNTRIMMED_REFERENCE_CONFIG:
        return UNTRIMMED_REFERENCE, "text_trim off, FA4 off at both sites, torch VAE"
    return None, (f"profile {profile!r} with overrides {overrides} resolved to {config}: neither "
                  f"{SERVED_DEFAULT!r} (profile {SERVED_PROFILE!r}, no override, text_trim on) nor "
                  f"{UNTRIMMED_REFERENCE!r} ({UNTRIMMED_REFERENCE_CONFIG})")


def dims_mismatch(fixture_dims: dict[str, object], resolved_dims: dict[str, object]) -> dict[str, tuple]:
    """Keys of the fixture's fp16-reference dims the resolved dims disagree on.

    ``{key: (fixture, resolved)}``. Only the fixture's own keys are compared
    (a manifest predates keys such as ``num_views`` / ``image_h``), so the
    frontend built by ``load_imagewam`` runs the dims the reference was
    recorded with.
    """
    return {key: (value, resolved_dims.get(key)) for key, value in fixture_dims.items()
            if resolved_dims.get(key) != value}


def seed_source(*, stamp: str, git: dict[str, object], device_name: str, capability: tuple[int, int],
                clock_state: dict[str, object], warmup: int, iters: int, fixture: str,
                effective_config: str) -> str:
    """The ``source`` text of a seeded baseline: what it was measured under."""
    commit = str(git.get("commit"))[:12]
    dirty = "" if git.get("clean") else " (worktree not clean)"
    clocks = (f"nvpmodel {clock_state.get('nvpmodel_mode')}, gpu_locked={clock_state.get('gpu_locked')}, "
              f"emc_locked={clock_state.get('emc_locked')}")
    return (f"gate run {stamp}, commit {commit}{dirty}, {device_name} sm_{capability[0]}{capability[1]}, "
            f"{clocks}, --warmup {warmup} --iters {iters}, fixture {fixture}; {effective_config}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    """HEAD plus the worktree state, untracked files included.

    ``tracked_changes`` covers modified tracked files only; untracked
    files are listed separately (first 50) with their count, and
    ``clean`` is true only when there are neither.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": f"unknown ({type(exc).__name__})", "tracked_changes": None,
                "untracked_files": None, "untracked_count": None, "clean": None}
    tracked = [line[3:] for line in status if not line.startswith("??")]
    untracked = [line[3:] for line in status if line.startswith("??")]
    return {"commit": head, "tracked_changes": bool(tracked), "tracked_changed_files": tracked[:50],
            "untracked_files": untracked[:50], "untracked_count": len(untracked), "clean": not status}


def explicit_constructor_params() -> set[str]:
    """Keywords ``ImageWAMTorchFrontendThor.__init__`` declares by name (not ``**kwargs``)."""
    return {name for name, p in inspect.signature(ImageWAMTorchFrontendThor.__init__).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def text_trim_mismatch(manifest: FixtureManifest, text_trim: bool) -> str | None:
    """Compare the fixture's trimming switch with the configuration under test.

    The gate's ``fp16`` reference was recorded with the fixture's own
    ``text_trim`` (``manifest.text_trim``; a manifest without the field is
    untrimmed, which is fixture v1). A run of the other value is a
    different computation, so it is refused rather than measured against a
    reference of the other kind. Returns a reason on mismatch, ``None``
    when the two agree.
    """
    if manifest.text_trim == text_trim:
        return None
    return (f"fixture {manifest.name} was recorded with text_trim={manifest.text_trim}, the "
            f"configuration under test runs text_trim={text_trim}: the fixture's fp16 reference is "
            f"compared against a run of the same switch (ISSUE-080); use a fixture recorded "
            f"text_trim={text_trim} (benchmarks/imagewam_gate_fixture_generate.py)")


def verify_checkpoint(ckpt: str, stats: str, manifest: FixtureManifest, skip_hash: bool,
                      context: dict[str, object]) -> str | None:
    """Check the checkpoint and dataset stats against the fixture; return a reason on mismatch.

    The checkpoint is compared by SHA-256 when the manifest has one;
    ``skip_hash`` replaces that with a byte-size comparison. How it was
    verified goes into ``context["checkpoint"]``.
    """
    expected = manifest.metadata["checkpoint"]
    record = context["checkpoint"]
    expected_sha = expected.get("sha256")
    if skip_hash:
        record["verified_by"] = "size (--skip-checkpoint-hash)"
        if os.path.getsize(ckpt) != expected["bytes"]:
            return f"checkpoint is {os.path.getsize(ckpt)} bytes, fixture's is {expected['bytes']}"
    elif expected_sha is None:
        return "fixture manifest has no checkpoint sha256; pass --skip-checkpoint-hash to accept a size-only check"
    else:
        start = time.time()
        actual_sha = file_sha256(Path(ckpt))
        record.update(sha256=actual_sha, verified_by="sha256", hash_s=round(time.time() - start, 1))
        if actual_sha != expected_sha:
            return f"checkpoint sha256 {actual_sha[:16]} != fixture's {expected_sha[:16]}"
    expected_stats = manifest.metadata["dataset_stats"]["sha256"]
    actual_stats = file_sha256(Path(stats))
    if actual_stats != expected_stats:
        return f"dataset_stats.json sha256 {actual_stats[:16]} != fixture's {expected_stats[:16]}"
    return None


def context_tensors(fixture: ImageWAMGateFixture, task: int) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.from_numpy(fixture.context_bf16_bits[task].view(np.int16).copy())
    return bits.view(BF16), torch.from_numpy(fixture.context_mask[task].copy())


def observation(fixture: ImageWAMGateFixture, index: int) -> dict[str, object]:
    return {"view1": torch.from_numpy(fixture.view1[index].copy()),
            "view2": torch.from_numpy(fixture.view2[index].copy()),
            "proprio": fixture.state[index]}


def masked_mae(pred_real: np.ndarray, gt: np.ndarray, gt_len: int) -> float:
    return float(np.abs(pred_real[:gt_len] - gt[:gt_len]).mean())


def run_fidelity(fe: ImageWAMTorchFrontendThor, fixture: ImageWAMGateFixture,
                 action_norm: MinMaxNormalizer) -> tuple[FidelityMeasurement, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    current_task = None
    all_finite = True
    for i in range(fixture.num_observations):
        task = int(fixture.task_index[i])
        if task != current_task:
            ctx, mask = context_tensors(fixture, task)
            fe.set_prompt(context=ctx, context_mask=mask)
            current_task = task
        obs = observation(fixture, i)
        for j, seed in enumerate(fixture.seeds.tolist()):
            noise = torch.from_numpy(fixture.noise[i, j].copy()).to(DEV)
            real = fe.infer(obs, action_noise=noise)["actions"]
            normalized = action_norm.forward(torch.from_numpy(real)).cpu()
            vs_official = parity_metrics(normalized, torch.from_numpy(fixture.official_actions[i, j]))
            vs_fp16 = parity_metrics(normalized, torch.from_numpy(fixture.fp16_reference_actions[i, j]))
            reference_real = action_norm.backward(
                torch.from_numpy(fixture.fp16_reference_actions[i, j])).cpu().numpy()
            gt_len = int(fixture.gt_len[i])
            finite = bool(np.isfinite(real).all())
            all_finite = all_finite and finite
            rows.append({
                "episode": int(fixture.episode[i]), "frame": int(fixture.frame[i]), "seed": int(seed),
                "vs_official": vs_official["cosine"], "vs_official_max_abs": vs_official["max_abs"],
                "vs_fp16_reference": vs_fp16["cosine"], "vs_fp16_reference_max_abs": vs_fp16["max_abs"],
                "mae_vs_gt": masked_mae(real, fixture.gt_actions[i], gt_len),
                "reference_mae_vs_gt": masked_mae(reference_real, fixture.gt_actions[i], gt_len),
                "finite": finite,
            })
    measurement = FidelityMeasurement(
        vs_official=CosineSummary.from_values([r["vs_official"] for r in rows]),
        vs_fp16_reference=CosineSummary.from_values([r["vs_fp16_reference"] for r in rows]),
        mae_vs_gt_mean=float(np.mean([r["mae_vs_gt"] for r in rows])),
        reference_mae_vs_gt_mean=float(np.mean([r["reference_mae_vs_gt"] for r in rows])),
        all_finite=all_finite)
    return measurement, rows


def run_latency(fe: ImageWAMTorchFrontendThor, obs: dict[str, object], warmup: int, iters: int) -> LatencySummary:
    """Served ``infer()`` as deployed (default noise); ``infer()`` synchronizes before returning."""
    for _ in range(warmup):
        fe.infer(obs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fe.infer(obs)
        samples.append((time.perf_counter() - start) * 1e3)
    return LatencySummary.from_samples(samples)


def by_seed(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        subset = [r for r in rows if r["seed"] == seed]
        vs_official = [r["vs_official"] for r in subset]
        out[f"seed_{seed}"] = {"vs_official_median": float(np.median(vs_official)),
                               "vs_official_min": float(np.min(vs_official)),
                               "vs_fp16_reference_median": float(np.median([r["vs_fp16_reference"] for r in subset])),
                               "mae_vs_gt_mean": float(np.mean([r["mae_vs_gt"] for r in subset]))}
    return out


def emit(report: GateReport, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = report.to_dict()
    (output_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(RESULT_PREFIX + json.dumps(record, sort_keys=True), flush=True)
    print(f"verdict: {report.verdict} ({report.reason})")
    print(f"latency: {report.latency} ({report.latency_reason})")
    if report.latency_seed is not None:
        print("latency_seed (paste `entry` at `path` in the baselines file): "
              + json.dumps(report.latency_seed, sort_keys=True))
    for check in report.checks:
        print(f"  {check.status:8s} {check.name:26s} value={check.value} limit={check.limit}  {check.detail}")
    print(f"result: {output_dir / 'result.json'}", flush=True)
    return report.exit_code


def record_configuration(fe: ImageWAMTorchFrontendThor, context: dict[str, object]) -> None:
    """Record what the frontend resolved and is running into ``context["config"]``.

    ``use_fa4`` / ``use_fa4_mot`` are the runtime-resolved values (FA4 where
    the machine can run it; both False after a capture-time fallback,
    ``fa4_fallback_reason`` then says why); ``effective_config`` is the
    line ``format_effective_config`` prints for them.
    """
    config = context["config"]
    config["use_fa4"] = fe.use_fa4   # what the frontend resolved on this machine
    config["use_fa4_mot"] = fe.use_fa4_mot
    config["fa4_fallback_reason"] = fe.fa4_fallback_reason
    config["effective_config"] = format_effective_config(
        fe.resolved_config.options, use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
        fa4_fallback_reason=fe.fa4_fallback_reason)


def build_parser() -> argparse.ArgumentParser:
    """The command line, whose defaults are the served configuration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--fixture-dir", type=Path, default=os.environ.get(FIXTURE_DIR_ENV),
                        help=f"directory holding {FIXTURE_FILE} (default ${FIXTURE_DIR_ENV})")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="fixture manifest; defaults to the served fixture "
                             "(imagewam_libero_gate_v2, trimmed), while the untrimmed reference is "
                             "imagewam_libero_gate_v1")
    parser.add_argument("--profile", default=SERVED_PROFILE,
                        help="config_resolver profile the frontend is built from (default: the "
                             "served `default`)")
    parser.add_argument("--text-trim", action=argparse.BooleanOptionalAction, default=True,
                        help="the expert text_trim override: run the configuration under test with "
                             "text_trim=True, the served default; --no-text-trim runs it untrimmed "
                             "(the untrimmed reference also needs FA4 off, see --override). Either "
                             "way it must equal the fixture's own text_trim")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="an expert key of the resolver (config_resolver.EXPERT_KEYS), repeatable; "
                             "true/false are bools, integers ints, anything else a string, and "
                             "use_fa4/use_fa4_mot/vae_graph also take `auto`. A run with an override "
                             "is not the served_default configuration; text_trim has its own flag")
    parser.add_argument("--thresholds", type=Path, default=CONFIG_DIR / "fidelity_thresholds.json")
    parser.add_argument("--baselines", type=Path, default=CONFIG_DIR / "latency_baselines.json")
    parser.add_argument("--fp8-calibration", type=Path, default=os.environ.get(FP8_CALIBRATION_ENV))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-latency", action="store_true",
                        help="an ungated latency (no policy or no baseline) makes the verdict blocked")
    parser.add_argument("--skip-checkpoint-hash", action="store_true",
                        help="verify the checkpoint by byte size only, not by the manifest's SHA-256")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.fixture_dir is None:
        parser.error(f"--fixture-dir is required (or set ${FIXTURE_DIR_ENV})")
    if args.iters < LATENCY_GROUP_COUNT:
        parser.error(f"--iters must be at least {LATENCY_GROUP_COUNT} (latency group medians), got {args.iters}")
    if args.warmup < 0:
        parser.error(f"--warmup must be non-negative, got {args.warmup}")
    try:
        overrides = parse_overrides(args.override)
    except ValueError as exc:
        parser.error(str(exc))
    ae_model_path = os.environ.get("FLUX2_AE_MODEL_PATH", os.environ.get("AE_MODEL_PATH"))
    if ae_model_path is None:
        parser.error("FLUX2_AE_MODEL_PATH (or AE_MODEL_PATH) is required: the gate runs the real VAE")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path(f"/tmp/imagewam-gate-{args.precision}-{stamp}")

    device_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    policy = LatencyPolicyTable.load(args.baselines).resolve(device_name, capability)
    clock_state = report_jetson_clock_state()
    manifest = FixtureManifest.read(args.manifest)
    ckpt = os.environ["CKPT_PATH"]
    stats = os.path.join(os.path.dirname(ckpt), "dataset_stats.json")
    context: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "device_name": device_name, "compute_capability": list(capability),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "flash_rt_kernels_so": str(Path(fvk.__file__).resolve()),
        "flash_rt_kernels_sha256": file_sha256(Path(fvk.__file__)),
        "clock_state": clock_state.to_dict(),
        "fixture": {"name": manifest.name, "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": file_sha256(args.manifest), "dir": str(args.fixture_dir),
                    "file_sha256": manifest.files[FIXTURE_FILE].sha256,
                    "text_trim": manifest.text_trim},
        "checkpoint": {"path": ckpt, "bytes": os.path.getsize(ckpt)},
        "latency_policy": {"device": policy.device, "gated": policy.gated, "reason": policy.reason},
        # FA4 is not set from here: the frontend resolves it (the served
        # default). `None` is "not resolved yet"; the value the frontend
        # resolved replaces it after construction.
        "config": {"warmup": args.warmup, "iters": args.iters, "use_fa4": None,
                   "text_trim": args.text_trim, "require_latency": args.require_latency,
                   "profile": args.profile, "overrides": overrides},
    }

    mismatch = text_trim_mismatch(manifest, args.text_trim)
    if mismatch is not None:
        return emit(GateReport.not_run(args.precision, policy.device, VERDICT_BLOCKED, mismatch,
                                       context), output_dir)
    thresholds = FidelityThresholdTable.load(args.thresholds).for_precision(args.precision)
    if thresholds is None:
        return emit(GateReport.not_run(args.precision, policy.device, VERDICT_BLOCKED,
                                       f"no fidelity thresholds for precision {args.precision!r} in {args.thresholds}",
                                       context), output_dir)
    checkpoint_mismatch = verify_checkpoint(ckpt, stats, manifest, args.skip_checkpoint_hash, context)
    if checkpoint_mismatch is not None:
        return emit(GateReport.not_run(args.precision, policy.device, VERDICT_BLOCKED, checkpoint_mismatch,
                                       context), output_dir)

    frontend_kwargs: dict[str, object] = {}
    if thresholds.requires_calibration:
        calibration = args.fp8_calibration
        if calibration is None or not Path(calibration).is_file():
            return emit(GateReport.not_run(
                args.precision, policy.device, VERDICT_SKIPPED,
                f"{args.precision} requires a calibration file (--fp8-calibration or ${FP8_CALIBRATION_ENV}); "
                f"none found at {calibration}", context), output_dir)
        context["calibration"] = {"path": str(calibration), "sha256": file_sha256(Path(calibration))}
        if FP8_CALIBRATION_FRONTEND_KWARG not in explicit_constructor_params():
            return emit(GateReport.not_run(
                args.precision, policy.device, VERDICT_BLOCKED,
                f"calibration file present but ImageWAMTorchFrontendThor.__init__ declares no "
                f"{FP8_CALIBRATION_FRONTEND_KWARG!r} keyword", context), output_dir)
        frontend_kwargs[FP8_CALIBRATION_FRONTEND_KWARG] = str(calibration)

    fixture = GateFixtureStore(args.fixture_dir).load(manifest)
    dims = dict(manifest.metadata["fp16_reference"]["dims"])
    context["config"]["dims"] = dims
    start = time.time()
    try:
        # The deployment entry on the LIBERO workload: the profile and the
        # expert overrides are resolved (and refused when illegal) before
        # anything is allocated. `text_trim` is the --text-trim flag; the
        # overrides cannot carry it (parse_override).
        fe = load_imagewam(
            ckpt, ImageWAMWorkload.libero(), profile=args.profile, precision=args.precision,
            ae_model_path=ae_model_path, flux2_src=os.environ["FLUX2_SRC"], dataset_stats_path=stats,
            text_trim=args.text_trim, **overrides, **frontend_kwargs)
    except (RuntimeError, ValueError) as exc:
        return emit(GateReport.not_run(args.precision, policy.device, VERDICT_BLOCKED,
                                       f"frontend construction failed: {type(exc).__name__}: {exc}",
                                       context), output_dir)
    context["construct_s"] = round(time.time() - start, 1)
    record_configuration(fe, context)
    print(context["config"]["effective_config"], flush=True)
    changed = dims_mismatch(dims, fe.resolved_config.dims)
    if changed:
        return emit(GateReport.not_run(
            args.precision, policy.device, VERDICT_BLOCKED,
            f"the frontend resolved dims that differ from the fixture's fp16 reference dims "
            f"(fixture, resolved): {changed}", context), output_dir)
    fp4 = sys.modules.get("flash_rt.flash_rt_fp4")
    if fp4 is not None:
        context["flash_rt_fp4_sha256"] = file_sha256(Path(fp4.__file__))

    _, action_norm = load_real_normalizers(stats, device=DEV)
    measurement, rows = run_fidelity(fe, fixture, action_norm)
    context["fidelity"] = {"vs_official": asdict(measurement.vs_official),
                           "vs_fp16_reference": asdict(measurement.vs_fp16_reference),
                           "mae_vs_gt_mean": measurement.mae_vs_gt_mean,
                           "reference_mae_vs_gt_mean": measurement.reference_mae_vs_gt_mean,
                           "by_seed": by_seed(rows), "samples": rows}
    latency = run_latency(fe, observation(fixture, fixture.num_observations - 1), args.warmup, args.iters)
    context["latency"] = asdict(latency)
    context["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # FA4 can fall back to the cuBLAS chain at the first capture, so the
    # configuration is read again after the runs: what actually ran.
    record_configuration(fe, context)
    print(context["config"]["effective_config"], flush=True)
    config = resolved_configuration(fe.resolved_config.options, use_fa4=fe.use_fa4,
                                    use_fa4_mot=fe.use_fa4_mot)
    configuration, why = baseline_configuration(args.profile, overrides, config)
    context["config"]["baseline_configuration"] = configuration
    context["config"]["baseline_configuration_reason"] = why
    latency_gate = LatencyGate(policy)
    latency_check = latency_gate.evaluate(args.precision, latency, configuration, config,
                                          configuration_reason=why)
    seed = latency_gate.seed_entry(
        args.precision, configuration, latency, config, resolved_config=config,
        source=seed_source(stamp=stamp, git=context["git"], device_name=device_name,
                           capability=capability, clock_state=context["clock_state"],
                           warmup=args.warmup, iters=args.iters, fixture=manifest.name,
                           effective_config=context["config"]["effective_config"]))

    checks = FidelityGate(thresholds).evaluate(measurement) + [latency_check]
    return emit(GateReport.evaluated(args.precision, policy.device, checks, context,
                                     require_latency=args.require_latency, latency_seed=seed), output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

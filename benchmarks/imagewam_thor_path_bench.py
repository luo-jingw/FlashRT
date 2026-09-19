#!/usr/bin/env python
"""One ImageWAM workload, three service paths, one latency table.

Builds one frontend through the deployment entry `load_imagewam`
(plan.md "Plan: configuration consolidation", W11/W12) and measures the same
served tick three ways, for any `ImageWAMWorkload`:

  infer   `ImageWAMTorchFrontendThor.infer(observation)` -- the Python serving
          path: VAE/proprio preparation, then the captured graph replay
          (`benchmarks/imagewam_thor_graph_bench.py`,
          `tests/gate_imagewam_libero.py`).
  abi     `fe.export_model_runtime(io="python")` adopted by
          `tests/_helpers/model_runtime_consumer.ModelRuntimeConsumer`; one
          tick writes the image tokens, the proprio row and the initial
          noise, calls `step()` and reads `actions`
          (`tests/gate_imagewam_model_runtime_export.py`).
  native  `fe.runtime_surface()` -> `ImageWAMNativeRuntime.create(surface)` ->
          `native.use_graph(surface.graph_exec)` (the graph the Python
          frontend captured, replayed from the native handle:
          `tests/gate_imagewam_native_parity.py --graph python`) ->
          `fe.export_model_runtime(io="native", native=native)`. The graph
          producer the native handle reports is printed.

Each path is measured the way the gates measure (warmup, then
`torch.cuda.synchronize()` on both sides of a wall-clock timer) and reported
as P10/P50/P90 and n. A path that cannot be built in this process is reported
as skipped with the reason and does not stop the run: `exec/build` (Python
ABI) and `runtime/build` + the `flashrt_imagewam_native` target (native face)
are separate builds (docs/imagewam_model_runtime.md), and the ABI and native
faces refuse configurations the Python path accepts (rules R5/R6: `text_trim`
with a single-graph consumer, FA4 and the in-graph VAE on the native face, so
`--profile fast` measures `infer` only).

The numbers are LATENCY ONLY. With no `CKPT_PATH` the weights are random (the
model structure is then `ImageWAMStructure.libero()`, the real 4B release
table; with `CKPT_PATH` it is read from the checkpoint -- the bench reads it
once, for the derived layout and the context width, and hands that same
structure to `load_imagewam` instead of letting it read the file again). The
frames are seeded random uint8 and the text context is a random BF16 tensor
(`_imagewam_workload_cli.random_observation` / `random_context`), so no LIBERO
data and no Qwen3 text encoder is involved: nothing here is an accuracy,
fidelity or end-to-end number. `FLUX2_AE_MODEL_PATH` (or `AE_MODEL_PATH`) and
`FLUX2_SRC`, a pair the constructor takes together or not at all, are only
needed when the resolved configuration runs a real FLUX2
VAE -- the native VAE encoder or the VAE inside the graph (rule R3, e.g.
`--profile fast`); without them `img_raw` keeps the placeholder image tokens
`infer()` refills and the graph reads, which changes no kernel any path here
runs.

The frontend's observation path carries the workload's own view count
(`observation_views` reads `view1` ... `view<num_views>`, `stage_images`
takes that many views in either VAE placement), so a three-view workload
drives the in-graph VAE stage through `infer()` with all three views.
Without a real VAE the views are not consumed at all (`stage_inputs`
refills `img_raw`), so all three paths measure a three-view workload with
random weights and frames.

Prints the workload and its derived layout, the resolved `effective_config`
line, the Jetson clock state (read-only, `flash_rt.hardware.jetson_clock_state`),
one line per path, and one paste-able `__IMAGEWAM_PATH_BENCH__` summary line
per path.

    # LIBERO workload, random weights, all three paths (Thor)
    python benchmarks/imagewam_thor_path_bench.py

    # the candidate target workload (3 x 256x256, horizon 32), one profile
    python benchmarks/imagewam_thor_path_bench.py --workload target --profile fast

    # the recorded LIBERO numbers with the real checkpoint and the fast profile
    CKPT_PATH=$CKPT_PATH python benchmarks/imagewam_thor_path_bench.py --profile fast --precision nvfp4

Env: CKPT_PATH (weights and structure; `dataset_stats.json` beside it),
FLUX2_SRC, FLUX2_AE_MODEL_PATH or AE_MODEL_PATH, QWEN3_MODEL_SPEC -- the same
names the ImageWAM gates and `benchmarks/imagewam_e2e_official_compare.py`
read. `--profile`, `--precision` and the switch flags are expert overrides on
top of a named profile (`config_resolver.PROFILES`, `EXPERT_KEYS`): an unset
flag leaves the profile's own value in place, so `--profile fast` alone runs
the whole `fast` profile. `use_fa4` is `None` in `default`, which resolves at
construction from `FLASHRT_THOR_FA4`; `--use-fa4 on|off` is the explicit
override. A static-FP8 precision needs a calibration file (rule R1) and this
bench takes none, so `--precision fp8_static*` raises the resolver's `R1`
before anything is allocated -- build one with
`benchmarks/imagewam_build_calibration.py` and run the comparison through
`benchmarks/imagewam_e2e_official_compare.py`'s `CALIBRATION`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

# `benchmarks/` (this script's own helper) and `tests/` (the gates' `_helpers`)
# on sys.path, so the two imports below work both when this file runs as a
# script and when it is imported as a module. The gates rely on the script
# directory being sys.path[0] for the same import; this makes it explicit.
_BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(os.path.dirname(_BENCHMARKS_DIR), "tests")
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path  # noqa: E402
from _imagewam_workload_cli import (  # noqa: E402
    add_workload_args, random_context, random_observation, view_frames, workload_from_args,
)

from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state
from flash_rt.models.imagewam.config_resolver import (
    PROFILES, VAE_ENCODERS, Precision, format_effective_config,
)
from flash_rt.models.imagewam.workload import ImageWAMWorkload, SequenceLayout

if TYPE_CHECKING:
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

PATHS = ("infer", "abi", "native")
SUMMARY_PREFIX = "__IMAGEWAM_PATH_BENCH__ "


@dataclass(frozen=True)
class Percentiles:
    """P10 / P50 / P90 of one path's per-tick wall time, in ms."""
    p10: float
    p50: float
    p90: float
    n: int

    @staticmethod
    def of(ms: list[float]) -> "Percentiles":
        q = np.percentile(ms, (10, 50, 90))
        return Percentiles(p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), n=len(ms))

    def line(self) -> str:
        return f"P10={self.p10:.2f} P50={self.p50:.2f} P90={self.p90:.2f} ms (n={self.n})"


@dataclass(frozen=True)
class PathResult:
    """What one requested service path produced: numbers, or the reason it
    could not be built in this process."""
    path: str
    measured: bool
    percentile: Percentiles | None
    reason: str


@dataclass(frozen=True)
class TickInputs:
    """The three inputs one ABI / native tick writes, and the chunk it reads
    back: the image tokens as raw host bits (`bits(fe._img_raw)`, the
    `image_tokens` SWAP window is `img_raw` itself), the camera frames the
    `image_views` window stacks when the VAE runs inside the graph, the
    proprio bytes (`set_input("proprio")`, f32) and the initial action latent
    (`noise`, the window `actions_raw` also reads)."""
    image_tokens: np.ndarray
    frames: list[np.ndarray]
    proprio: bytes
    noise: np.ndarray
    chunk: tuple[int, int]


def format_workload(workload: ImageWAMWorkload) -> str:
    """The nine workload fields as `key=value` pairs."""
    return (f"num_views={workload.num_views} image_h={workload.image_h} image_w={workload.image_w} "
            f"text_max_len={workload.text_max_len} action_horizon={workload.action_horizon} "
            f"action_dim={workload.action_dim} proprio_dim={workload.proprio_dim} "
            f"num_steps={workload.num_steps} shift={workload.shift}")


def format_layout(layout: SequenceLayout) -> str:
    """The derived sequence layout as `key=value` pairs."""
    return (f"x0={layout.x0} img_len={layout.img_len} a0={layout.a0} total={layout.total} "
            f"ref_h={layout.ref_h} ref_w={layout.ref_w} dt={layout.dt:.6g}")


def parse_paths(value: str) -> tuple[str, ...]:
    """`--paths` as the requested path names, in the order given."""
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in PATHS]
    if unknown or not names:
        raise ValueError(f"--paths {value!r}: want a comma-separated subset of {PATHS}")
    return names


def expert_overrides(args: argparse.Namespace) -> dict[str, bool | str]:
    """The `**expert` pack for `load_imagewam` (`config_resolver.EXPERT_KEYS`):
    only the switches actually passed, so an unset flag leaves the profile's
    own value. `--text-trim`, `--use-fa4-mot` and `--vae-graph` are on|off and
    `--vae-encoder` is one of `config_resolver.VAE_ENCODERS`."""
    expert: dict[str, bool | str] = {}
    if args.text_trim is not None:
        expert["text_trim"] = args.text_trim == "on"
    if args.use_fa4 is not None:
        expert["use_fa4"] = args.use_fa4 == "on"
    if args.use_fa4_mot is not None:
        expert["use_fa4_mot"] = args.use_fa4_mot == "on"
    if args.vae_encoder is not None:
        expert["vae_encoder"] = args.vae_encoder
    if args.vae_graph is not None:
        expert["vae_graph"] = args.vae_graph == "on"
    return expert


def image_token_bits(fe: ImageWAMTorchFrontendThor) -> np.ndarray:
    """The frontend's image tokens as raw host bits, exactly what the gates
    write into the `image_tokens` SWAP window
    (`tests/_helpers/imagewam_abi_checks.bits(fe._img_raw)`).

    `_helpers.imagewam_abi_checks` imports `flash_rt.models.imagewam.runtime_export`,
    which needs the `exec/` build; importing it here, inside a path's own
    `try`, turns a missing build into that path's skip reason.
    """
    from _helpers.imagewam_abi_checks import bits
    return bits(fe._img_raw)


def make_tick(consumer: ModelRuntimeConsumer, inputs: TickInputs) -> Callable[[], None]:
    """The gates' verb sequence for one ABI / native tick.

    The image input is the `image_tokens` SWAP window when the face has that
    port (the VAE outside the graph; both gates' bench ticks write it) and the
    `image_views` SWAP window otherwise (the VAE inside the graph, which takes
    every view at once).
    """
    names = {port.name for port in consumer.ports}
    if "image_tokens" in names:
        def write_image() -> None:
            consumer.write_swap("image_tokens", inputs.image_tokens)
    elif "image_views" in names:
        views = np.stack(inputs.frames)
        def write_image() -> None:
            consumer.write_swap("image_views", views)
    else:
        raise ValueError(f"no image input port among {sorted(names)}")

    def tick() -> None:
        write_image()
        consumer.set_input("proprio", inputs.proprio)
        consumer.write_swap("noise", inputs.noise)
        consumer.step()
        consumer.get_output("actions", np.float32, inputs.chunk)

    return tick


def time_ms(tick: Callable[[], None], warmup: int, iters: int) -> Percentiles:
    """`warmup` untimed ticks, then `iters` ticks timed with
    `torch.cuda.synchronize()` on both sides of the wall clock -- the gates'
    measurement (so an `infer()` number here is the same quantity
    `tests/gate_imagewam_libero.py` reports)."""
    for _ in range(warmup):
        tick()
    torch.cuda.synchronize()
    ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tick()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - start) * 1e3)
    return Percentiles.of(ms)


def bench_infer(fe: ImageWAMTorchFrontendThor, observation: dict[str, object], noise: torch.Tensor,
                *, warmup: int, iters: int) -> Percentiles:
    """The Python serving path: `infer()` on the random observation, with the
    same initial action latent the ABI and native ticks write."""
    def tick() -> None:
        fe.infer(observation, action_noise=noise)

    return time_ms(tick, warmup, iters)


def bench_abi(fe: ImageWAMTorchFrontendThor, inputs_factory: Callable[[], TickInputs],
              *, warmup: int, iters: int) -> Percentiles:
    """The `io="python"` ABI: the frontend's own graph and staging verbs behind
    `frt_model_runtime_v1`, driven through a ctypes consumer."""
    mr = fe.export_model_runtime(io="python", identity={"bench": "imagewam_thor_path_bench"})
    consumer: ModelRuntimeConsumer | None = None
    try:
        consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
        print(f"  abi: ports={[port.name for port in consumer.ports]} stages={consumer.n_stages} "
              f"fingerprint=0x{consumer.fingerprint:016x}")
        return time_ms(make_tick(consumer, inputs_factory()), warmup, iters)
    finally:
        if consumer is not None:
            consumer.close()
        mr.release()


def bench_native(fe: ImageWAMTorchFrontendThor, inputs_factory: Callable[[], TickInputs],
                 *, warmup: int, iters: int) -> Percentiles:
    """The native face: `frt_imagewam_native` over the runtime surface, with
    the graph the Python frontend captured installed by `use_graph` and the
    C verbs installed on the `io="native"` declaration."""
    from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime

    surface = fe.runtime_surface()
    native = ImageWAMNativeRuntime.create(surface)
    try:
        native.use_graph(surface.graph_exec)
        print(f"  native: graph_producer={native.graph_producer} nodes={native.graph_nodes} "
              f"view_shape={surface.view_shape}")
        mr = fe.export_model_runtime(io="native", native=native,
                                     identity={"bench": "imagewam_thor_path_bench"})
        consumer: ModelRuntimeConsumer | None = None
        try:
            consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
            print(f"  native: ports={[port.name for port in consumer.ports]} "
                  f"fingerprint=0x{consumer.fingerprint:016x}")
            return time_ms(make_tick(consumer, inputs_factory()), warmup, iters)
        finally:
            if consumer is not None:
                consumer.close()
            mr.release()
    finally:
        native.close()


def run_path(path: str, fe: ImageWAMTorchFrontendThor, observation: dict[str, object],
             noise: torch.Tensor, workload: ImageWAMWorkload, *, warmup: int, iters: int) -> PathResult:
    """Measure one requested path. Everything a path needs is built inside its
    own `try`, so a missing native build or a configuration one face refuses
    becomes that path's skip reason instead of ending the run."""

    def inputs_factory() -> TickInputs:
        return TickInputs(image_tokens=image_token_bits(fe),
                          frames=view_frames(observation, workload.num_views),
                          proprio=np.asarray(observation["proprio"], dtype=np.float32).tobytes(),
                          noise=noise.cpu().numpy(),
                          chunk=(int(fe.dims["num_action"]), int(fe.dims["action_dim"])))

    try:
        if path == "infer":
            percentile = bench_infer(fe, observation, noise, warmup=warmup, iters=iters)
        elif path == "abi":
            percentile = bench_abi(fe, inputs_factory, warmup=warmup, iters=iters)
        else:
            percentile = bench_native(fe, inputs_factory, warmup=warmup, iters=iters)
    except Exception as e:  # a path that cannot be built here is a skip, not a failure
        reason = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        print(f"{path:<7} SKIPPED ({reason})")
        return PathResult(path=path, measured=False, percentile=None, reason=reason)
    print(f"{path:<7} {percentile.line()}")
    return PathResult(path=path, measured=True, percentile=percentile, reason="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workload_args(ap)
    ap.add_argument("--profile", default="default", choices=tuple(PROFILES),
                    help="named profile (config_resolver.PROFILES)")
    ap.add_argument("--precision", default=None, choices=[tier.value for tier in Precision],
                    help="overrides the profile's precision")
    ap.add_argument("--paths", default=",".join(PATHS),
                    help=f"comma-separated subset of {PATHS}")
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the random frames, proprio, context and initial action noise")
    ap.add_argument("--text-trim", choices=("on", "off"), default=None)
    ap.add_argument("--use-fa4", choices=("on", "off"), default=None)
    ap.add_argument("--use-fa4-mot", choices=("on", "off"), default=None)
    ap.add_argument("--vae-encoder", choices=VAE_ENCODERS, default=None)
    ap.add_argument("--vae-graph", choices=("on", "off"), default=None)
    args = ap.parse_args()

    if args.bench_iters <= 0 or args.warmup < 0:
        ap.error("--bench-iters must be > 0 and --warmup >= 0")
    try:
        paths = parse_paths(args.paths)
    except ValueError as e:
        ap.error(str(e))

    from flash_rt.frontends.torch.imagewam_thor import load_imagewam
    from flash_rt.models.imagewam.structure import ImageWAMStructure

    ckpt = os.environ.get("CKPT_PATH")
    ae_model_path = os.environ.get("FLUX2_AE_MODEL_PATH") or os.environ.get("AE_MODEL_PATH")
    # The constructor takes `ae_model_path` and `flux2_src` together or not at
    # all (imagewam_thor.py:306), so `flux2_src` stays unset when there is no
    # autoencoder: without one the workload runs on the placeholder image
    # tokens, which is the measured path here. `load_real_ae` puts `flux2_src`
    # on `sys.path` itself, so nothing else has to.
    flux2_src = os.environ.get("FLUX2_SRC") if ae_model_path else None
    dataset_stats_path = os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None

    workload = workload_from_args(args)
    # Read once and hand to `load_imagewam` (which would read the same file
    # again for `structure=None`): the derived layout and the context width
    # (`joint_attention_dim`) come from the structure the frontend is built on.
    structure = ImageWAMStructure.libero() if ckpt is None else ImageWAMStructure.from_checkpoint(ckpt)
    layout = workload.layout(structure)

    print(f"workload: {format_workload(workload)}")
    print(f"layout:   {format_layout(layout)}")

    started = time.time()
    fe = load_imagewam(ckpt, workload, structure=structure, profile=args.profile, precision=args.precision,
                       ae_model_path=ae_model_path, flux2_src=flux2_src,
                       qwen3_model_spec=os.environ.get("QWEN3_MODEL_SPEC"),
                       dataset_stats_path=dataset_stats_path, consumer="infer",
                       **expert_overrides(args))
    resolved = fe.resolved_config
    if resolved is None:
        raise RuntimeError("load_imagewam did not record a resolved configuration")
    print(f"frontend: profile={args.profile} precision={resolved.options.precision.value} "
          f"weights={'real' if ckpt else 'random'} constructed in {time.time() - started:.1f}s")
    print(format_effective_config(resolved.options, use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
                                  fa4_fallback_reason=fe.fa4_fallback_reason))

    report_jetson_clock_state()

    context, context_mask = random_context(workload, int(fe.dims["joint_attention_dim"]), args.seed)
    fe.set_prompt(context=context, context_mask=context_mask)
    torch.cuda.synchronize()
    print(f"ready in {time.time() - started:.1f}s; active context rows x0={fe.active_dims['x0']} "
          f"(dims x0={fe.dims['x0']}, device {torch.cuda.get_device_name()})")

    observation = random_observation(workload, args.seed)
    torch.manual_seed(args.seed)
    noise = torch.empty_like(fe._action_latent).normal_().mul_(0.01)
    print(f"inputs: seed={args.seed}, {workload.num_views} random "
          f"{workload.image_h}x{workload.image_w} frames, random context "
          f"({workload.text_max_len} x {fe.dims['joint_attention_dim']}), 0.01 * N(0, 1) action latent "
          f"-- latency only, no LIBERO data and no Qwen3")

    print(f"\npaths (warmup {args.warmup}, {args.bench_iters} timed ticks each):")
    results = [run_path(path, fe, observation, noise, workload, warmup=args.warmup, iters=args.bench_iters)
               for path in paths]

    print("\nsummary (one line per path):")
    for result in results:
        fields = [f"path={result.path}", f"status={'ok' if result.measured else 'skipped'}",
                  f"profile={args.profile}", f"precision={resolved.options.precision.value}",
                  f"workload={args.workload}", format_workload(workload), format_layout(layout)]
        if result.percentile is None:
            fields.append(f"reason={result.reason!r}")
        else:
            fields.append(f"P10={result.percentile.p10:.2f} P50={result.percentile.p50:.2f} "
                          f"P90={result.percentile.p90:.2f} n={result.percentile.n}")
        print(SUMMARY_PREFIX + " ".join(fields))

    measured = [result for result in results if result.measured]
    if not measured:
        print("no requested path could be built in this process", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Kernel-level profile of one captured ImageWAM graph, per text length.

Answers "which kernels do not get smaller when `text_trim` makes the sequence
shorter?". The 0921 Thor round measured `precision="fp16"` at 274.7 ms with
`text_trim` on (`x0 = 25`) against 275 ms untrimmed, while `nvfp4` drops from
202 to 103 ms with the same switch, so some part of the fp16 graph does not
scale with the rows. This script times the graph and lists, for each requested
valid-token count, where the GPU time goes.

For every (precision, valid tokens) pair it builds the frontend through
`load_imagewam` (no autoencoder: the transformer graph only), sets a random
context of that many valid tokens (with `text_trim` the graph is captured at
`x0 = valid + 1`; 512 valid tokens is the untrimmed shape), replays the graph
under `torch.profiler` and prints

  * the graph replay time (CUDA events) and the sum of the kernel times inside
    it (the difference is idle gaps between kernels),
  * the kernel count per replay,
  * the GPU time by category (GEMM, attention, elementwise/other) and the
    top kernels by total time, with their per-replay count and time.

    CKPT_PATH=... python benchmarks/imagewam_graph_kernel_profile.py \\
        --precision fp16 --valid-tokens 24,512

Run it for `--precision fp16` and `--precision nvfp4`, and diff the tables: a
kernel whose per-replay time is the same at 24 and 512 valid tokens is the part
that does not scale. Random weights unless CKPT_PATH is set (latency does not
depend on the values). `--use-fa4 on|off|auto` selects FA4 at both sites;
`off` removes FA4 as a variable.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _imagewam_workload_cli import (  # noqa: E402
    add_workload_args, random_context, workload_from_args,
)

CATEGORIES = (
    ("gemm", ("gemm", "cutlass", "cublas", "xmma", "nvjet", "sm80_", "sm90_", "sm100_", "sm110_", "wgmma")),
    ("attention", ("flash", "fmha", "softmax", "attn", "attention", "mot_joint", "perhead")),
)


def category(name: str) -> str:
    low = name.lower()
    for label, needles in CATEGORIES:
        if any(n in low for n in needles):
            return label
    return "elementwise/other"


def _time_total(evt) -> float:
    """Device time in microseconds across torch versions."""
    for attr in ("device_time_total", "cuda_time_total"):
        value = getattr(evt, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _is_kernel(evt) -> bool:
    return _time_total(evt) > 0 and "cuda" in str(getattr(evt, "device_type", "cuda")).lower()


def profile_graph(graph, replays: int) -> tuple[list[tuple[str, int, float]], float]:
    """`(kernels, replay_ms)`: per kernel `(name, count per replay, total us per replay)` and the mean
    graph replay time from CUDA events."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    replay_ms = start.elapsed_time(end) / replays

    # A throwaway session first: the first profiler session of a process initialises CUPTI and can
    # record no kernels at all (the 0921d run got 0 kernels for each precision's first graph).
    with profile(activities=[ProfilerActivity.CUDA]):
        graph.replay()
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        if not _is_kernel(evt):
            continue
        rows.append((evt.key, evt.count / replays, _time_total(evt) / replays))
    return rows, replay_ms


def report(label: str, rows: list[tuple[str, int, float]], replay_ms: float, top: int) -> None:
    kernel_us = sum(us for _, _, us in rows)
    count = sum(n for _, n, _ in rows)
    print(f"\n== {label}")
    print(f"graph replay {replay_ms:.2f} ms; kernels inside {kernel_us / 1e3:.2f} ms "
          f"({100 * kernel_us / 1e3 / replay_ms:.0f}% of the replay), {count:.0f} kernel launches")
    by_cat: dict[str, float] = {}
    for name, _, us in rows:
        by_cat[category(name)] = by_cat.get(category(name), 0.0) + us
    print("by category: " + ", ".join(f"{c} {us / 1e3:.2f} ms" for c, us in sorted(by_cat.items())))
    # How much of the GPU time is in kernels too short to be limited by the tensor cores or the memory
    # system (launch / tail latency dominates them): the share and the average kernel duration.
    print(f"average kernel {kernel_us / max(count, 1):.1f} us; " + "; ".join(
        f"kernels < {cut} us: {sum(n for _, n, us in rows if us / max(n, 1) < cut):.0f} launches, "
        f"{sum(us for _, n, us in rows if us / max(n, 1) < cut) / 1e3:.2f} ms "
        f"({100 * sum(us for _, n, us in rows if us / max(n, 1) < cut) / max(kernel_us, 1):.0f}%)"
        for cut in (10, 25, 100)))
    print(f"top {top} kernels (per replay):")
    for name, n, us in sorted(rows, key=lambda r: -r[2])[:top]:
        print(f"  {us / 1e3:8.3f} ms  x{n:<6.0f} {category(name):<18} {name[:110]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workload_args(ap)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--valid-tokens", default="24,512",
                    help="comma-separated valid-token counts; 512 is the untrimmed shape for LIBERO")
    ap.add_argument("--use-fa4", choices=("on", "off", "auto"), default="off")
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from flash_rt.frontends.torch.imagewam_thor import load_imagewam
    from flash_rt.models.imagewam.config_resolver import format_effective_config
    from flash_rt.models.imagewam.structure import ImageWAMStructure

    ckpt = os.environ.get("CKPT_PATH")
    workload = workload_from_args(args)
    structure = ImageWAMStructure.libero() if ckpt is None else ImageWAMStructure.from_checkpoint(ckpt)
    expert = {} if args.use_fa4 == "auto" else {"use_fa4": args.use_fa4 == "on",
                                                "use_fa4_mot": args.use_fa4 == "on"}
    counts = [int(v) for v in args.valid_tokens.split(",")]
    print(f"precision={args.precision} profile={args.profile} workload: {workload}")
    for valid in counts:
        fe = load_imagewam(ckpt, workload, structure=None if ckpt else structure, profile=args.profile,
                           precision=args.precision, dataset_stats_path=(
                               os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None),
                           **expert)
        ctx, mask = random_context(workload, structure.joint_attention_dim, args.seed, valid_tokens=valid)
        fe.set_prompt(context=ctx, context_mask=mask)
        rows, replay_ms = profile_graph(fe._graph, args.replays)
        config = format_effective_config(
            fe.resolved_config.options, use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
            fa4_fallback_reason=fe.fa4_fallback_reason)
        report(f"{args.precision}, valid_tokens={valid}, {config}", rows, replay_ms, args.top)
        del fe
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())

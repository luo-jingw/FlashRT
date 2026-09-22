#!/usr/bin/env python
"""Where the GPU time goes: computation-graph stage x operator class.

The frontend runs one eager pass (`run_eager`: the same kernels in the same
order as the captured graph) under `torch.profiler`, with every pipeline stage
wrapped in a named range (encode, each backbone double / single layer, the
prefill remainder, each ActionDiT double / single layer, the denoise-step
remainder). The trace is parsed here: every kernel is attributed to the
innermost stage range that launched it (through the launch correlation id, so
it needs no GPU-side annotation support) and to an operator class by its name.
The output is

  * the grid: rows = stages, columns = operator classes, cells = GPU ms (the
    whole pass, all layers and all denoise steps of that stage summed),
  * the kernel launches per stage and class and the average kernel duration,
  * the share of GPU time in kernels shorter than 10 / 25 us per stage (too
    short for the tensor cores or the memory system to be the limit: launch and
    tail latency dominate them),
  * for the GEMM class of each layer stage, the achieved TFLOPs and the weight
    bytes streamed per second, computed from the model's own dimensions
    (approximate: the action double blocks are counted as one stream),
    against which to read "compute bound" (TFLOPs near the tensor-core rate)
    or "transport bound" (GB/s near the ~250 GB/s of the memory system).

Eager kernels have the same durations as the graph's; the gaps between them are
not measured here (in the graph the kernels cover ~100% of the replay,
`imagewam_graph_kernel_profile.py`).

    CKPT_PATH=... python benchmarks/imagewam_stage_operator_grid.py \\
        --precision nvfp4 --valid-tokens 24

Random weights unless CKPT_PATH is set. No autoencoder is built, so the VAE is
outside the grid. `--use-fa4 off` (default) removes FA4 as a variable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager

STAGE_PREFIX = "stage:"
CLASSES = ("gemm", "quantize", "norm", "rope", "attention", "glu", "residual", "copy", "other")
# (class, name needles) in match order: the first class with a needle in the kernel name wins.
NEEDLES = (
    # quantize checked before attention: this project's own kernels live in a `flash_rt::fp4`
    # namespace, so "flash" alone would misclassify them as attention (0921x found
    # kernel_quantize_fp4_sfa_vec counted under "attention" for exactly this reason).
    ("quantize", ("quant",)),
    ("attention", ("fmha", "flash_attn", "flashattention", "attn", "attention", "softmax",
                   "mot_joint", "perhead")),
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "xmma", "wgmma", "sm80_", "sm90_", "sm100_", "sm110_")),
    ("norm", ("adaln", "layernorm", "layer_norm", "rmsnorm", "rms_norm", "norm")),
    ("rope", ("rope",)),
    ("glu", ("silu", "glu", "gelu", "swiglu")),
    ("residual", ("residual", "gated", "gate", "add_")),
    ("copy", ("copy", "memcpy", "memset", "transpose", "cast", "convert", "slice", "fill")),
)
SHORT_CUTS_US = (10, 25)


def classify(name: str) -> str:
    low = name.lower()
    for cls, needles in NEEDLES:
        if any(n in low for n in needles):
            return cls
    return "other"


def attribute(trace: dict) -> list[tuple[str, str, str, float]]:
    """`(stage, class, kernel name, duration us)` for every GPU kernel / memcpy / memset of a Chrome
    trace, the stage being the innermost `stage:*` CPU range that contains the kernel's launch (by
    correlation id); `"(none)"` when no stage range does."""
    events = trace["traceEvents"]
    launch_ts: dict = {}
    ranges = []
    for e in events:
        cat = e.get("cat")
        if cat == "cuda_runtime" and "correlation" in e.get("args", {}):
            launch_ts[e["args"]["correlation"]] = e["ts"]
        elif cat in ("user_annotation", "cpu_op") and str(e.get("name", "")).startswith(STAGE_PREFIX):
            ranges.append((e["ts"], e["ts"] + e.get("dur", 0.0), e["name"][len(STAGE_PREFIX):]))
    out = []
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        ts = launch_ts.get(e.get("args", {}).get("correlation"))
        stage = "(none)"
        if ts is not None:
            inside = [(end - start, label) for start, end, label in ranges if start <= ts <= end]
            if inside:
                stage = min(inside)[1]
        cls = "copy" if e["cat"] != "kernel" else classify(e["name"])
        out.append((stage, cls, e["name"], float(e.get("dur", 0.0))))
    return out


def build_grid(rows: list[tuple[str, str, str, float]]) -> dict:
    """stage -> {"us": {cls: us}, "n": {cls: launches}, "short": {cut: us}, "names": {name: us}}."""
    grid: dict = defaultdict(lambda: {"us": defaultdict(float), "n": defaultdict(int),
                                      "short": defaultdict(float), "names": defaultdict(float)})
    for stage, cls, name, us in rows:
        g = grid[stage]
        g["us"][cls] += us
        g["n"][cls] += 1
        g["names"][name] += us
        for cut in SHORT_CUTS_US:
            if us < cut:
                g["short"][cut] += us
    return grid


def _params(dims: dict) -> dict:
    """Weight parameters per block type from the model dimensions (analytic, approximate)."""
    h, m = dims["hidden"], dims["mlp_hidden"]
    ah, aw, am = dims["action_hidden_dim"], dims["action_attn_width"], dims["action_mlp_hidden"]
    dbl_stream = 3 * h * h + h * h + h * 2 * m + m * h
    sgl = h * (3 * h + 2 * m) + (h + m) * h
    a_dbl = ah * 3 * aw + aw * ah + ah * 2 * am + am * ah
    a_sgl = ah * (3 * aw + 2 * am) + (aw + am) * ah
    return dict(dbl_stream=dbl_stream, sgl=sgl, a_dbl=a_dbl, a_sgl=a_sgl)


def analytic(dims: dict, bytes_per_param: float) -> dict:
    """stage -> (gemm FLOPs, weight bytes read) of one full pass, from the dimensions."""
    p = _params(dims)
    nd, ns = dims["num_layers_double"], dims["num_layers_single"]
    and_, ans = dims["action_num_layers_double"], dims["action_num_layers_single"]
    x0, a0, na, steps = dims["x0"], dims["a0"], dims["num_action"], dims["num_denoise_steps"]
    img = a0 - x0
    return {
        "bb.double": (2 * nd * (x0 + img) * p["dbl_stream"], nd * 2 * p["dbl_stream"] * bytes_per_param),
        "bb.single": (2 * ns * a0 * p["sgl"], ns * p["sgl"] * bytes_per_param),
        "act.double": (2 * steps * and_ * na * p["a_dbl"], steps * and_ * p["a_dbl"] * bytes_per_param),
        "act.single": (2 * steps * ans * na * p["a_sgl"], steps * ans * p["a_sgl"] * bytes_per_param),
    }


def render(grid: dict, dims: dict, bytes_per_param: float) -> str:
    stages = sorted(grid, key=lambda s: -sum(grid[s]["us"].values()))
    total_us = sum(sum(g["us"].values()) for g in grid.values())
    lines = [f"\nGPU time by stage x operator class (ms; whole eager pass; total {total_us / 1e3:.1f} ms)"]
    head = f"{'stage':<20}{'total':>8}{'launch':>8}{'avg us':>8} | " + " ".join(f"{c:>9}" for c in CLASSES)
    lines += [head, "-" * len(head)]
    for s in stages:
        g = grid[s]
        us, n = sum(g["us"].values()), sum(g["n"].values())
        lines.append(f"{s:<20}{us / 1e3:>8.2f}{n:>8}{us / max(n, 1):>8.1f} | "
                     + " ".join(f"{g['us'].get(c, 0.0) / 1e3:>9.2f}" for c in CLASSES))
    col = {c: sum(g["us"].get(c, 0.0) for g in grid.values()) / 1e3 for c in CLASSES}
    lines.append(f"{'ALL':<20}{total_us / 1e3:>8.2f}{sum(sum(g['n'].values()) for g in grid.values()):>8}{'':>8} | "
                 + " ".join(f"{col[c]:>9.2f}" for c in CLASSES))
    lines.append("\nkernels shorter than 10 / 25 us: share of the stage's GPU time")
    for s in stages:
        g = grid[s]
        us = max(sum(g["us"].values()), 1e-9)
        lines.append(f"  {s:<20}{100 * g['short'].get(10, 0.0) / us:>5.0f}% / {100 * g['short'].get(25, 0.0) / us:>4.0f}%")
    ref = analytic(dims, bytes_per_param)
    lines.append("\nGEMM class of the layer stages: achieved rate (analytic FLOPs and weight bytes / measured GEMM time)")
    for s, (flops, wbytes) in ref.items():
        gemm_us = grid.get(s, {"us": {}})["us"].get("gemm", 0.0)
        if gemm_us:
            lines.append(f"  {s:<14}{flops / 1e9:>9.1f} GFLOP {wbytes / 1e9:>7.2f} GB weights  "
                         f"-> {flops / gemm_us / 1e6:>7.1f} TFLOPs, {wbytes / gemm_us / 1e3:>7.1f} GB/s")
    return "\n".join(lines)


def unclassified(grid: dict, top: int) -> list[tuple[float, str]]:
    seen: dict[str, float] = defaultdict(float)
    for g in grid.values():
        for name, us in g["names"].items():
            if classify(name) == "other":
                seen[name] += us
    return sorted(((us, n) for n, us in seen.items()), reverse=True)[:top]


@contextmanager
def stage_ranges():
    """Wrap the pipeline's stage functions in `record_function("stage:<name>")`, restore on exit."""
    from torch.profiler import record_function

    from flash_rt.frontends.torch import imagewam_thor as ft
    from flash_rt.models.imagewam import pipeline_thor as pt

    # (module, function name, stage label): the layer functions and the encode / step functions are
    # looked up in pipeline_thor's globals at call time; the frontend imported `imagewam_prefill` and
    # `imagewam_denoise_loop` by name, so those two are wrapped in its namespace.
    targets = [(pt, "imagewam_encode_once", "encode"), (pt, "_double_stream_layer", "bb.double"),
               (pt, "_single_stream_layer", "bb.single"), (pt, "_action_double_layer", "act.double"),
               (pt, "_action_single_layer", "act.single"), (pt, "imagewam_denoise_step", "act.step_rest"),
               (ft, "imagewam_prefill", "bb.prefill_rest"), (ft, "imagewam_denoise_loop", "act.loop_rest")]
    original = [(mod, name, getattr(mod, name)) for mod, name, _ in targets]

    def wrap(fn, label):
        def wrapped(*a, **k):
            with record_function(STAGE_PREFIX + label):
                return fn(*a, **k)
        return wrapped

    for (mod, name, label), (_, _, fn) in zip(targets, original):
        setattr(mod, name, wrap(fn, label))
    try:
        yield
    finally:
        for mod, name, fn in original:
            setattr(mod, name, fn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _imagewam_workload_cli import add_workload_args, random_context, workload_from_args

    add_workload_args(ap)
    ap.add_argument("--precision", default="nvfp4")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--valid-tokens", type=int, default=24)
    ap.add_argument("--use-fa4", choices=("on", "off", "auto"), default="off")
    ap.add_argument("--calibration", default=None, help="calibration file for a static-FP8 precision")
    ap.add_argument("--show-unclassified", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    from flash_rt.frontends.torch.imagewam_thor import load_imagewam
    from flash_rt.models.imagewam.precision import Precision
    from flash_rt.models.imagewam.structure import ImageWAMStructure

    ckpt = os.environ.get("CKPT_PATH")
    workload = workload_from_args(args)
    structure = ImageWAMStructure.libero() if ckpt is None else ImageWAMStructure.from_checkpoint(ckpt)
    expert = {} if args.use_fa4 == "auto" else {"use_fa4": args.use_fa4 == "on", "use_fa4_mot": args.use_fa4 == "on"}
    fe = load_imagewam(ckpt, workload, structure=None if ckpt else structure, profile=args.profile,
                       precision=args.precision, calibration_path=args.calibration, dataset_stats_path=(
                           os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None), **expert)
    ctx, mask = random_context(workload, structure.joint_attention_dim, args.seed, valid_tokens=args.valid_tokens)
    fe.set_prompt(context=ctx, context_mask=mask)
    print(f"precision={args.precision} valid_tokens={args.valid_tokens} active dims x0={fe.active_dims['x0']} "
          f"a0={fe.active_dims['a0']}")
    for _ in range(2):
        fe.run_eager()
    torch.cuda.synchronize()
    # A throwaway profiler session first: the first `torch.profiler` session of a process
    # initializes CUPTI and can record zero kernels (0921x: all four attempts here came back
    # "the trace holds no GPU kernels", the same failure imagewam_graph_kernel_profile.py's own
    # X4 fixed the same way -- this script's version of the fix was missing until now).
    with profile(activities=[ProfilerActivity.CUDA]):
        fe.run_eager()
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof, stage_ranges():
        fe.run_eager()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            rows = attribute(json.load(f))
    if not rows:
        print("the trace holds no GPU kernels: torch.profiler did not record CUDA activity here", file=sys.stderr)
        return 1
    grid = build_grid(rows)
    bpp = {"fp16": 2.0, "fp16_cutlass": 2.0, "fp8": 1.0, "fp8_static": 1.0, "fp8_static_cutlass": 1.0}.get(
        Precision(args.precision).value, 0.5625)
    print(render(grid, fe.active_dims, bpp))
    other = unclassified(grid, args.show_unclassified)
    if other:
        print("\nkernels no class matched (us total, name): extend NEEDLES from these")
        for us, name in other:
            print(f"  {us:>9.1f}  {name[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

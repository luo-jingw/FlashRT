#!/usr/bin/env python
"""Text-context trimming (`text_trim=True`, issues.md ISSUE-020): capture
cost and device memory per text length, and trimmed vs full-length speed,
in one process at the real dims (x0=513 max with proprio, a0=905,
64 actions, 10 steps).

Sections (`--section`, default `all`):

  capture  one frontend with text_trim=True; `set_prompt` once per
           valid-token count in `--lengths` (default: the 15 distinct
           LIBERO counts over the four suites), then once more per count.
           New length: wall time of `set_prompt` (autotune of the shapes it
           adds on fp16, warmup, capture; the first one also runs the
           max-dims scratch prefill and the calibration) and the device
           memory it adds. Cached length: wall time of `set_prompt`.
  ab       the same frontend, `--ab-valid` valid tokens (trimmed) vs 512
           valid tokens (x0=513, the untrimmed shapes), alternating:
           graph replay alone (CUDA events) and `infer()` (wall clock,
           synchronized; placeholder image tokens unless --vae-graph).

Memory per new length is reported three ways: torch
`memory_reserved()` delta (the graph's private pool and any autotune
scratch torch still caches), this process's NVML used memory delta
(`nvidia-smi --query-compute-apps`, when available; includes the CUDA
graph's own driver allocations), and the device-wide
`cudaMemGetInfo` free-memory delta (includes other processes).

Weights are random unless CKPT_PATH is set (latency and memory do not
depend on the values). `--vae-graph` puts the real VAE stage in every
graph (needs FLUX2_SRC and AE_MODEL_PATH or FLUX2_AE_MODEL_PATH).
`--use-fa4 on|off|auto` (auto = `FLASHRT_THOR_FA4=1` opt-in rule) and
`--use-fa4-mot on|off` select FA4 per site (Thor only).

On a shared GPU the timings are indicative only. Prints one
`__TEXT_TRIM_BENCH__ <json>` line per result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

DEV = "cuda"
BF16 = torch.bfloat16
RESULT_PREFIX = "__TEXT_TRIM_BENCH__ "
LIBERO_VALID_COUNTS = (16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
TEXT_LEN = 512
REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=TEXT_LEN + 1, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=0.1, num_denoise_steps=10,
    ref_h=14, ref_w=28, proprio_dim=8, shift=5.0, num_train_timesteps=1000,
)


@dataclass(frozen=True)
class MemorySnapshot:
    torch_reserved: int
    torch_allocated: int
    process_used: int | None   # NVML, this process
    device_free: int


@dataclass(frozen=True)
class Percentiles:
    p10: float
    p50: float
    p90: float

    @staticmethod
    def of(values: list[float]) -> "Percentiles":
        return Percentiles(*(float(np.percentile(values, q)) for q in (10, 50, 90)))


def _process_used_bytes() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pid = str(os.getpid())
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0] == pid and fields[1].isdigit():
            return int(fields[1]) * 2**20
    return None


def snapshot() -> MemorySnapshot:
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    return MemorySnapshot(torch.cuda.memory_reserved(), torch.cuda.memory_allocated(), _process_used_bytes(), free)


def _mib(delta: int | None) -> float | None:
    return None if delta is None else round(delta / 2**20, 1)


def emit(kind: str, **fields: object) -> None:
    print(RESULT_PREFIX + json.dumps(dict(kind=kind, **fields)), flush=True)


def _flux2_src() -> str:
    src = os.environ["FLUX2_SRC"]
    return os.path.join(src, "src") if os.path.isdir(os.path.join(src, "src", "flux2")) else src


def build_frontend(args: argparse.Namespace) -> ImageWAMTorchFrontendThor:
    ckpt = os.environ.get("CKPT_PATH")
    stats = os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None
    vae = {}
    if args.vae_graph:
        vae = dict(ae_model_path=os.environ.get("AE_MODEL_PATH") or os.environ["FLUX2_AE_MODEL_PATH"],
                   flux2_src=_flux2_src(), vae_graph_input=(2, 224, 224))
    use_fa4 = {"on": True, "off": False, "auto": None}[args.use_fa4]
    t = time.perf_counter()
    fe = ImageWAMTorchFrontendThor(precision=args.precision, dims_override=dict(REAL_DIMS), ckpt_path=ckpt,
                                   dataset_stats_path=stats, text_trim=True, use_fa4=use_fa4,
                                   use_fa4_mot=args.use_fa4_mot == "on", **vae)
    emit("construct", seconds=round(time.perf_counter() - t, 2), precision=args.precision,
         weights="real" if ckpt else "random", use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
         vae_graph=bool(args.vae_graph), device=torch.cuda.get_device_name())
    return fe


def _context_and_mask(n_valid: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(n_valid)
    ctx = torch.randn(TEXT_LEN, REAL_DIMS["joint_attention_dim"], generator=g).to(BF16)
    mask = torch.zeros(TEXT_LEN, dtype=torch.bool)
    mask[:n_valid] = True
    return ctx, mask


def _observation(args: argparse.Namespace) -> dict[str, object]:
    obs: dict[str, object] = {"proprio": np.zeros(REAL_DIMS["proprio_dim"], dtype=np.float32)}
    if args.vae_graph:
        g = torch.Generator().manual_seed(0)
        obs["view1"] = torch.randint(0, 256, (224, 224, 3), generator=g, dtype=torch.uint8)
        obs["view2"] = torch.randint(0, 256, (224, 224, 3), generator=g, dtype=torch.uint8)
    return obs


def section_capture(fe: ImageWAMTorchFrontendThor, counts: list[int]) -> None:
    print(f"\n=== capture: {len(counts)} lengths (valid tokens {counts}) ===", flush=True)
    for n in counts:
        ctx, mask = _context_and_mask(n)
        before = snapshot()
        t = time.perf_counter()
        fe.set_prompt(context=ctx, context_mask=mask)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        after = snapshot()
        process = (None if before.process_used is None or after.process_used is None
                   else after.process_used - before.process_used)
        emit("new_length", n_valid=n, x0=fe.active_dims["x0"], a0=fe.active_dims["a0"],
             seconds=round(seconds, 3), torch_reserved_mib=_mib(after.torch_reserved - before.torch_reserved),
             torch_allocated_mib=_mib(after.torch_allocated - before.torch_allocated),
             process_used_mib=_mib(process), device_free_drop_mib=_mib(before.device_free - after.device_free))
    for n in counts:
        ctx, mask = _context_and_mask(n)
        t = time.perf_counter()
        fe.set_prompt(context=ctx, context_mask=mask)
        torch.cuda.synchronize()
        emit("cached_length", n_valid=n, x0=fe.active_dims["x0"], seconds=round(time.perf_counter() - t, 4))
    emit("captures", lengths=list(fe.captured_text_lengths),
         torch_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2),
         peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def _replay_ms(graph: torch.cuda.CUDAGraph, iters: int) -> list[float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))
    return out


def _infer_ms(fe: ImageWAMTorchFrontendThor, obs: dict[str, object], iters: int) -> list[float]:
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fe.infer(obs)
        out.append((time.perf_counter() - t) * 1e3)
    return out


def section_ab(fe: ImageWAMTorchFrontendThor, ab_valid: int, iters: int, rounds: int,
               obs: dict[str, object]) -> None:
    sides = {"trimmed": ab_valid, "full": TEXT_LEN}
    graphs = {}
    for name, n in sides.items():
        ctx, mask = _context_and_mask(n)
        fe.set_prompt(context=ctx, context_mask=mask)
        graphs[name] = (fe._graph, fe.active_dims["x0"], ctx, mask)
    print(f"\n=== ab: trimmed x0={graphs['trimmed'][1]} vs full x0={graphs['full'][1]}, "
          f"{rounds} rounds x {iters} iters, alternating ===", flush=True)
    replay: dict[str, list[float]] = {name: [] for name in sides}
    infer: dict[str, list[float]] = {name: [] for name in sides}
    for name in sides:  # warmup
        _replay_ms(graphs[name][0], 3)
    for _ in range(rounds):
        for name in sides:
            replay[name] += _replay_ms(graphs[name][0], iters)
        for name in sides:
            _, _, ctx, mask = graphs[name]
            fe.set_prompt(context=ctx, context_mask=mask)
            _infer_ms(fe, obs, 2)
            infer[name] += _infer_ms(fe, obs, iters)
    for name in sides:
        emit("ab", side=name, x0=graphs[name][1], replay_ms=asdict(Percentiles.of(replay[name])),
             infer_ms=asdict(Percentiles.of(infer[name])), samples=len(replay[name]))
    r = Percentiles.of(replay["trimmed"]).p50 / Percentiles.of(replay["full"]).p50
    i = Percentiles.of(infer["trimmed"]).p50 / Percentiles.of(infer["full"]).p50
    emit("ab_ratio", replay_p50_trimmed_over_full=round(r, 4), infer_p50_trimmed_over_full=round(i, 4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="all", choices=["all", "capture", "ab"])
    ap.add_argument("--precision", default="nvfp4")
    ap.add_argument("--lengths", default=",".join(str(n) for n in LIBERO_VALID_COUNTS),
                    help="valid-token counts for the capture section")
    ap.add_argument("--ab-valid", type=int, default=20, help="valid tokens on the trimmed side of the A/B")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--use-fa4", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--use-fa4-mot", default="off", choices=["on", "off"])
    ap.add_argument("--vae-graph", action="store_true")
    args = ap.parse_args()
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}", flush=True)
    fe = build_frontend(args)
    if args.section in ("all", "capture"):
        section_capture(fe, [int(v) for v in args.lengths.split(",")])
    if args.section in ("all", "ab"):
        section_ab(fe, args.ab_valid, args.iters, args.rounds, _observation(args))
    emit("done", fa4_fallback_reason=fe.fa4_fallback_reason)


if __name__ == "__main__":
    main()

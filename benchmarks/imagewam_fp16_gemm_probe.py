#!/usr/bin/env python
"""Per-shape timing of the fp16 cuBLASLt GEMM path the `fp16` tier runs on.

The 0921 Thor rounds measured `precision="fp16"` between 172 and 295 ms for the
same trimmed graph depending on the harness and the run (issues.md ISSUE-088),
while the CUTLASS tiers are stable, and `nsys` showed single `nvjet` kernels at
4.47 ms per call (a 47 GFLOP GEMM should take a few tenths of a millisecond).
`GemmRunner.fp16_nn` runs the algorithm `cublasLtMatmulAlgoGetHeuristic` ranks
first for a shape until `autotune_fp16_nn` replaces it with the fastest of the
top 16 measured on zero-filled scratch tensors. This probe times, for every
distinct weight-GEMM shape a graph at a given text length uses,

  * `default`  the heuristic's top-1 algorithm,
  * `zeros`    the algorithm `autotune_fp16_nn` picks when it is fed the zero
               tensors `imagewam_thor._autotune_gemm` feeds it,
  * `random`   the algorithm it picks when fed random tensors,

each re-timed on RANDOM operands with CUDA events, and repeats the whole thing
`--repeats` times with a fresh `GemmRunner` so a pick that differs between runs
shows up as a spread. Column `TFLOPs` is the achieved rate; a fp16 GEMM of these
sizes on Thor should be far above 20.

    python benchmarks/imagewam_fp16_gemm_probe.py --x0 25,513

Weights are random; nothing here needs a checkpoint. `--x0` lists the context
lengths (`x0 = valid tokens + 1`; 513 is the untrimmed shape); `a0 = x0 + 392`.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import flash_rt.flash_rt_kernels as fvk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS  # noqa: E402

FP16 = torch.float16
DEV = "cuda"


def shapes_for(x0: int, dims: dict) -> list[tuple[str, int, int, int]]:
    """The weight-GEMM shapes of `imagewam_thor._autotune_gemm` at context length `x0`."""
    hidden, mlp = dims["hidden"], dims["mlp_hidden"]
    img_len = dims["a0"] - dims["x0"]
    a0 = x0 + img_len
    ahd, aaw, amh, na = (dims["action_hidden_dim"], dims["action_attn_width"],
                         dims["action_mlp_hidden"], dims["num_action"])
    out = [
        ("txt_qkv", x0, 3 * hidden, hidden), ("txt_proj", x0, hidden, hidden),
        ("txt_mlp0", x0, mlp * 2, hidden), ("txt_mlp2", x0, hidden, mlp),
        ("img_qkv", img_len, 3 * hidden, hidden), ("img_proj", img_len, hidden, hidden),
        ("img_mlp0", img_len, mlp * 2, hidden), ("img_mlp2", img_len, hidden, mlp),
        ("single_qkv", a0, 3 * hidden, hidden), ("single_attn_out", a0, hidden, hidden),
        ("single_mlp_in", a0, mlp * 2, hidden), ("single_mlp_down", a0, hidden, mlp),
        ("single_linear2", a0, hidden, hidden + mlp),
        ("action_qkv", na, 3 * aaw, ahd), ("action_proj", na, ahd, aaw),
        ("action_mlp0", na, amh * 2, ahd), ("action_mlp2", na, ahd, amh),
        ("action_linear2", na, ahd, aaw + amh),
    ]
    return out


def time_ms(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def probe(m: int, n: int, k: int, autotune_data: str | None) -> float:
    """Time of one `fp16_nn` on random operands, on a FRESH runner: the heuristic's top-1
    (`autotune_data=None`) or the algorithm autotuned on zero / random operands."""
    runner = fvk.GemmRunner()
    a = torch.randn(m, k, device=DEV, dtype=FP16)
    w = torch.randn(k, n, device=DEV, dtype=FP16) * 0.02
    d = torch.zeros(m, n, device=DEV, dtype=FP16)
    if autotune_data is not None:
        if autotune_data == "zeros":
            za, zw, zd = (torch.zeros_like(t) for t in (a, w, d))
        else:
            za, zw, zd = a, w, d
        runner.autotune_fp16_nn(za.data_ptr(), zw.data_ptr(), zd.data_ptr(), m, n, k, 16)
    return time_ms(lambda: runner.fp16_nn(a.data_ptr(), w.data_ptr(), d.data_ptr(), m, n, k, 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", default="25,513", help="comma-separated context lengths")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dims = dict(LIBERO_REAL_DIMS)
    print(f"{torch.cuda.get_device_name(0)}; rows: shape, GFLOP, then ms (TFLOPs) per mode, "
          f"min..max over {args.repeats} fresh runners")
    seen: set[tuple[int, int, int]] = set()
    for x0 in (int(v) for v in args.x0.split(",")):
        print(f"\n== x0={x0} a0={x0 + dims['a0'] - dims['x0']}")
        print(f"{'shape':<16}{'M':>5}{'N':>6}{'K':>6}{'GFLOP':>8}  {'default ms (TFLOPs)':<26}"
              f"{'autotune@zeros':<28}{'autotune@random':<28}")
        for name, m, n, k in shapes_for(x0, dims):
            if (m, n, k) in seen:
                continue
            seen.add((m, n, k))
            gflop = 2.0 * m * n * k / 1e9
            cells = []
            for mode in (None, "zeros", "random"):
                ms = [probe(m, n, k, mode) for _ in range(args.repeats)]
                lo, hi = min(ms), max(ms)
                tf = gflop / lo
                cells.append(f"{lo:7.3f}..{hi:7.3f} ({tf:5.1f})")
            print(f"{name:<16}{m:>5}{n:>6}{k:>6}{gflop:>8.1f}  {cells[0]:<26}{cells[1]:<28}{cells[2]:<28}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

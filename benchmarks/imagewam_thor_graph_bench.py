#!/usr/bin/env python
"""Canonical real-frontend, all-precisions steady-state speed bench.

Measures CUDA-graph-captured whole-pipeline `infer()` latency via the
REAL `ImageWAMTorchFrontendThor` (not a hand-rolled standalone
approximation) at ImageWAM's real dims, looping over every precision
`_PRECISIONS` lists. This is now the CANONICAL place to compare
fp16/fp8/fp8_static/fp8_static_cutlass/fp16_cutlass/nvfp4 -- the
former separate `imagewam_thor_fp16_bench.py`/`_fp8_bench.py`/
`_fp4_bench.py` scripts were deprecated (2026-09-17, see their own
module docstrings) because they duplicated a hand-rolled per-layer
approximation that had drifted out of sync with `pipeline_thor.py`'s
real math (missing AdaLN modulation, gated residual, and the real
merged SiLU-GLU MLP structure entirely -- a GENERIC transformer-block
skeleton, not FLUX.2/ImageWAM's real one) and used a stale placeholder
image-token shape. Going through the real frontend means this bench
can never drift out of sync with the real per-layer math again --
whatever `pipeline_thor.py` actually does is what gets measured, by
construction.

Real dims (opportunities.md, confirmed 2026-09-15/16/17), imported as
the served `libero_dims.LIBERO_REAL_DIMS` rather than re-typed here: the
served `x0` is the real text context, 512 real tokens + 1 reserved
proprio row -- proprio CONDITIONING itself is not enabled here, that
width is just the real structural width; a pure speed bench doesn't need
real proprio/dataset stats. `a0` is `x0` plus the real LIBERO
dual-camera image tokens, 392 = 14x28 grid from the real confirmed
224x448 input; `num_action` is the real LIBERO action horizon (confirmed
via `_imagewam_thor_spec.py`), `num_denoise_steps` the real
`eval_num_inference_steps`. Going through the real frontend also
sidesteps a real quirk a hand-rolled `make_imagewam_attention_spec` call
would hit at this exact `a0` (905, odd): the "backbone" site's
`kernel="standard"` path requires an EVEN `kv_seq` -- the real frontend's
own attention-site configuration already handles this correctly
(confirmed: real Thor `infer()` runs fine at this exact `a0`), so there is
nothing to work around here.

INT8/INT4 (SM80 CUTLASS, opportunities.md OPT-007) are NOT included --
they have no real dispatch path in `imagewam_thor.py`/`pipeline_thor.py`
at all (never wired in as a real precision option, closed for Thor on
speed grounds regardless), so they still need their own standalone
benchmarks (`imagewam_thor_int8_bench.py`/`_int4_bench.py`), kept
separately up to date with the real per-layer math.
"""
from __future__ import annotations

import statistics

import numpy as np
import torch

from flash_rt.frontends.torch.imagewam_thor import _PRECISIONS, ImageWAMTorchFrontendThor
from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS

# The served dims minus the entries this random-weight speed run leaves at
# the frontend's own defaults: `ref_h`/`ref_w` (would select the real 2D
# image RoPE grid in place of the flat placeholder), `proprio_dim` (proprio
# conditioning), `shift` and `num_train_timesteps` (the real timestep
# schedule).
_DEFAULTED_DIM_KEYS = ("ref_h", "ref_w", "proprio_dim", "shift", "num_train_timesteps")
REAL_DIMS = {key: value for key, value in LIBERO_REAL_DIMS.items()
             if key not in _DEFAULTED_DIM_KEYS}

WARMUP, ITERS = 15, 50


def bench_one(precision: str):
    frontend = ImageWAMTorchFrontendThor(dims_override=dict(REAL_DIMS), precision=precision)
    frontend.set_prompt("bench")  # triggers the one-time CUDA Graph capture
    torch.cuda.synchronize()

    obs = {"image": np.zeros((4, 4, 3), dtype=np.uint8)}

    for _ in range(WARMUP):
        frontend.infer(obs)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        frontend.infer(obs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)], statistics.mean(times)


def main():
    report_jetson_clock_state()
    print(f"Real dims: {REAL_DIMS}\n")
    print(f"{'precision':22s} {'P50 (ms)':>10s} {'P90 (ms)':>10s} {'mean (ms)':>10s}")
    for precision in _PRECISIONS:
        try:
            p50, p90, mean = bench_one(precision)
            print(f"{precision:22s} {p50:10.3f} {p90:10.3f} {mean:10.3f}")
        except RuntimeError as e:
            print(f"{precision:22s} SKIP ({type(e).__name__}: {str(e)[:80]})")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

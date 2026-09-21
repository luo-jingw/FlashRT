#!/usr/bin/env python
"""Does a failed CUDA graph capture leave the default CUDA generator unusable?

The 0921 Thor pytest run (torch 2.9.1) ended with `torch.randn` raising
`Offset increment outside graph capture encountered unexpectedly` when a later
test built a frontend, after the FA4 tests had failed captures on purpose. That
error is the generator's "a capture is open" flag still being set. Whether a
failed capture leaves it set is a property of the torch build, so this probes
each way a capture can fail and reports, per way, whether `torch.randn` on the
device still works afterwards and whether one tiny successful capture repairs it.

    python scripts/probe_capture_generator_state.py

Prints one line per case; exit code 1 when any case leaves the generator
unusable. Read-only apart from allocating a few tiny tensors.
"""
from __future__ import annotations

import sys

import torch


def randn_works() -> str:
    try:
        torch.randn(4, device="cuda")
        return "ok"
    except Exception as exc:  # the error text is the finding
        return f"{type(exc).__name__}: {str(exc)[:100]}"


def repaired_by_a_capture() -> str:
    g = torch.cuda.CUDAGraph()
    x = torch.zeros(4, device="cuda")
    try:
        with torch.cuda.graph(g):
            _ = x + 1
    except Exception as exc:
        return f"the repair capture itself failed: {type(exc).__name__}"
    return randn_works()


def case_sync_inside_capture() -> None:
    x = torch.zeros(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = x + 1
        torch.cuda.synchronize()  # invalidates the capture


def case_python_error_inside_capture() -> None:
    x = torch.zeros(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = x + 1
        raise RuntimeError("probe: failure inside the capture body")


def case_rng_inside_capture_then_error() -> None:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = torch.randn(4, device="cuda")  # registers the graph with the generator
        raise RuntimeError("probe: failure after an RNG use inside the capture")


CASES = (case_sync_inside_capture, case_python_error_inside_capture, case_rng_inside_capture_then_error)


def main() -> int:
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    torch.randn(4, device="cuda")
    bad = 0
    for case in CASES:
        caller = torch.cuda.current_stream()
        try:
            case()
            outcome = "capture did not fail (case is not probing anything here)"
        except Exception as exc:
            outcome = f"capture failed: {type(exc).__name__}"
        torch.cuda.set_stream(caller)
        after = randn_works()
        line = f"{case.__name__}: {outcome}; randn afterwards: {after}"
        if after != "ok":
            bad += 1
            line += f"; after one successful capture: {repaired_by_a_capture()}"
        print(line, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

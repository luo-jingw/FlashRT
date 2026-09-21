"""Opt-in test-suite sentinel for the CUDA generator's capture state (issues.md ISSUE-087).

With `FLASHRT_GENERATOR_SENTINEL=1` every test is followed by one tiny
`normal_()` on the CUDA device. On the Thor's torch 2.9.1 a capture that fails
inside `capture_end` (a device sync inside it) leaves the default generator
believing a capture is still open, and every later random draw in the process
raises `Offset increment outside graph capture encountered unexpectedly`. The
sentinel names the FIRST test after which that state is set, repairs it with
one successful tiny capture so the tests after it are judged on their own, and
fails that test's teardown. Without the variable this file does nothing.

    FLASHRT_GENERATOR_SENTINEL=1 python -m pytest tests/test_imagewam_*.py -q
"""
from __future__ import annotations

import os

import pytest

_SIGNATURE = "Offset increment outside graph capture"


@pytest.fixture(autouse=True)
def _cuda_generator_sentinel(request):
    yield
    if os.environ.get("FLASHRT_GENERATOR_SENTINEL") != "1":
        return
    import torch

    if not torch.cuda.is_available():
        return
    try:
        torch.empty(1, device="cuda").normal_()
    except RuntimeError as exc:
        if _SIGNATURE not in str(exc):
            raise
        graph = torch.cuda.CUDAGraph()
        probe = torch.zeros(1, device="cuda")
        with torch.cuda.graph(graph):
            _ = probe + 1  # a successful capture resets the generator's flag
        pytest.fail(f"{request.node.nodeid} left the CUDA generator in the capturing state "
                    f"(repaired here by one successful capture)", pytrace=False)

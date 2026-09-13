"""ImageWAMTorchFrontendThor end-to-end wiring test (plan.md Phase 5).

set_prompt() once, then infer() repeatedly with varying random
observations -- the scenario plan.md's own Phase 5 Observation
section commits to. Random weights, small dims (see
imagewam_thor.py's own _DEFAULT_DIMS) -- wiring and CUDA Graph
replay correctness only, not accuracy or Thor performance.
"""
import time

import numpy as np
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor


def test_set_prompt_then_repeated_infer():
    frontend = ImageWAMTorchFrontendThor()

    frontend.set_prompt("pick up the red cup")
    # Second call with the same prompt must be a fast no-op (no recapture).
    graph_before = frontend._graph
    frontend.set_prompt("pick up the red cup")
    assert frontend._graph is graph_before, "set_prompt recaptured on an unchanged prompt"

    latencies = []
    last_actions = None
    for _ in range(5):
        t0 = time.perf_counter()
        result = frontend.infer({"image": np.zeros((4, 4, 3), dtype=np.uint8)})
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        actions = result["actions"]
        assert actions.shape == (frontend.dims["num_action"], frontend.dims["action_hidden_dim"])
        assert np.isfinite(actions).all(), "infer() produced NaN/Inf actions"
        if last_actions is not None:
            assert not np.array_equal(actions, last_actions), (
                "infer() returned identical actions across calls with varying "
                "random observations/noise -- graph replay may be reading stale "
                "buffers instead of the freshly-written ones")
        last_actions = actions

    p50_ms = sorted(latencies)[len(latencies) // 2] * 1000
    print(f"PASS: 5x infer() finite, varying, no recapture. P50={p50_ms:.2f}ms "
          f"(small structural-dry-run dims, not a Thor performance number)")


if __name__ == "__main__":
    test_set_prompt_then_repeated_infer()

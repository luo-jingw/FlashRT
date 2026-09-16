"""Real flow-matching inference schedule (opportunities.md OPT-009's
follow-up) -- `flash_rt.models.imagewam.scheduler.build_inference_schedule`
must match the real `imagewam.models.backbones.schedulers.scheduler_continuous.WanContinuousFlowMatchScheduler.build_inference_schedule`
exactly. Skips cleanly if the real `imagewam` package isn't importable
(a local RESOURCE, same skip convention as this project's other
real-checkpoint tests, inverted)."""
import os

import torch

DEV = "cuda"
_IMAGEWAM_SRC = "/home/ljw/projects/pi0.5/tmp/ImageWAM/src"
_IMAGEWAM_AVAILABLE = os.path.isdir(_IMAGEWAM_SRC)


def test_schedule_matches_toy_shift():
    """No real `imagewam` package needed -- checks the ported formula's
    own internal consistency (deltas sum to -1.0, i.e. sigma goes
    1->0 exactly) at an arbitrary shift/step count."""
    from flash_rt.models.imagewam.scheduler import build_inference_schedule

    timesteps, deltas = build_inference_schedule(7, shift=3.0, num_train_timesteps=500, device=DEV)
    assert timesteps.shape == (7,)
    assert deltas.shape == (7,)
    assert torch.isfinite(timesteps).all() and torch.isfinite(deltas).all()
    assert abs(deltas.sum().item() - (-1.0)) < 1e-5, "deltas must sum to -1.0 (sigma: 1 -> 0 exactly)"
    print("PASS: build_inference_schedule internally consistent at shift=3.0, 7 steps")


def test_schedule_matches_real_scheduler():
    if not _IMAGEWAM_AVAILABLE:
        import pytest
        pytest.skip(f"real imagewam package not present at {_IMAGEWAM_SRC} on this machine")

    import sys
    if _IMAGEWAM_SRC not in sys.path:
        sys.path.insert(0, _IMAGEWAM_SRC)
    from imagewam.models.backbones.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

    from flash_rt.models.imagewam.scheduler import build_inference_schedule

    # Real confirmed LIBERO release values (config.yaml): shift=5.0,
    # num_train_timesteps=1000, eval_num_inference_steps=10.
    ref = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    t_ref, d_ref = ref.build_inference_schedule(10, device=DEV, dtype=torch.float32)
    t_mine, d_mine = build_inference_schedule(10, shift=5.0, num_train_timesteps=1000, device=DEV)
    assert torch.allclose(t_ref, t_mine), f"timesteps mismatch: {t_ref} vs {t_mine}"
    assert torch.allclose(d_ref, d_mine), f"deltas mismatch: {d_ref} vs {d_mine}"
    print("PASS: build_inference_schedule matches the real WanContinuousFlowMatchScheduler exactly "
          "(shift=5.0, num_train_timesteps=1000, 10 steps -- real LIBERO release values)")


if __name__ == "__main__":
    test_schedule_matches_toy_shift()
    if not _IMAGEWAM_AVAILABLE:
        print(f"SKIPPED real-scheduler test: {_IMAGEWAM_SRC} not present")
    else:
        test_schedule_matches_real_scheduler()
    print("PASS")

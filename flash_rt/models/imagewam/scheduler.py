"""Real ImageWAM flow-matching inference schedule (shift-based,
non-uniform) -- opportunities.md OPT-009's follow-up, found while
scoping real closed-loop testing. Ported EXACTLY from the real
`imagewam.models.backbones.schedulers.scheduler_continuous.WanContinuousFlowMatchScheduler`
(`build_inference_schedule`/`step`/`_phi`, read directly, not
re-derived) -- confirmed a PLAIN SINGLE-STEP EULER integrator
(`sample + model_output * delta`), NOT a multi-step method like UniPC,
so this is a formula substitution into FlashRT's existing
`fvk.gpu_euler_step` call site, not a new integrator.

Real confirmed parameters for the LIBERO release
(`config.yaml`, both `video_scheduler`/`action_scheduler` blocks use
the SAME values for this release): `train_shift=5.0`, `infer_shift=5.0`,
`num_train_timesteps=1000`. `eval_num_inference_steps: 10` (top-level
key) -- the real evaluation step count for THIS release, NOT the
scheduler's own generic default of 20 seen in some example configs.
"""
from __future__ import annotations

import torch

DEV = "cuda"
F32 = torch.float32


def phi(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Real `WanContinuousFlowMatchScheduler._phi`, verbatim."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def build_inference_schedule(num_inference_steps: int, *, shift: float = 5.0,
                              num_train_timesteps: int = 1000,
                              device: str = DEV) -> tuple[torch.Tensor, torch.Tensor]:
    """Real `WanContinuousFlowMatchScheduler.build_inference_schedule`,
    verbatim. Returns `(timesteps, deltas)`, each `(num_inference_steps,)`
    float32:
      - `timesteps[step]`: this step's real conditioning timestep, in
        the scheduler's own `[0, num_train_timesteps]` domain -- convert
        to FlashRT's own `[0,1]` `action_timestep` convention via
        `timesteps[step] / num_train_timesteps` (matches the real
        `imagewam.py`'s own `_scheduler_timestep_to_unit` exactly).
      - `deltas[step]`: this step's real (non-uniform) Euler step size,
        used directly in place of a fixed `dt` in `fvk.gpu_euler_step`
        (real `step()` is `sample + model_output * delta`, the SAME
        formula that kernel already implements).
    """
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=F32)
    sigma_steps = phi(u_steps, shift)
    timesteps = sigma_steps[:-1] * float(num_train_timesteps)
    deltas = sigma_steps[1:] - sigma_steps[:-1]
    return timesteps, deltas

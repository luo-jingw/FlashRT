"""Served dims of the real `ImageWAM-FLUX.2-4B-LIBERO` release.

One definition for the calibration builder and the fidelity/study
scripts: FLUX.2-4B backbone widths, `x0 = 512` Qwen3 tokens + 1 proprio
row, `img_len = 392` (two 224x224 views -> 14x28 latent grid),
`num_action = 64`, 10-step shift schedule (`shift = 5.0`). Matches
`benchmarks/imagewam_e2e_official_compare.py`'s `REAL_DIMS`.
"""
from __future__ import annotations

LIBERO_HORIZON = 64
LIBERO_STEPS = 10
LIBERO_SHIFT = 5.0

LIBERO_REAL_DIMS: dict = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=LIBERO_HORIZON, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / LIBERO_STEPS, num_denoise_steps=LIBERO_STEPS,
    ref_h=14, ref_w=28, proprio_dim=8, shift=LIBERO_SHIFT, num_train_timesteps=1000,
)

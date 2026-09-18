"""Served dims of the real `ImageWAM-FLUX.2-4B-LIBERO` release, derived
from the workload and the structure (plan.md "configuration
consolidation", W8).

`LIBERO_REAL_DIMS` is the `dims` mapping that
`resolve_config(ImageWAMWorkload.libero(), ImageWAMStructure.libero())`
produces, so the calibration builder, the fidelity/study scripts and the
tests read one definition instead of typing the numbers again. The
workload contributes the sequence layout (`x0`, `img_len` through
`ref_h * ref_w`, `a0`, `total`, `num_action`, `proprio_dim`, `dt`,
`num_denoise_steps`, `shift`, `num_train_timesteps`) and `action_dim`;
the structure contributes the FLUX.2-4B backbone widths and the
action-expert dims.

LIBERO serves two 224x224 views, concatenated along the width into the
`14 x 28` latent grid; 512 text tokens plus one proprio row give
`x0 = 513`, `img_len = 392`, `a0 = 905` and `total = 969` at the
64-step horizon and the 10-step shift schedule (`shift = 5.0`). Matches
`benchmarks/imagewam_e2e_official_compare.py`'s `REAL_DIMS`.

`LIBERO_HORIZON`, `LIBERO_STEPS` and `LIBERO_SHIFT` stay module constants
for the benchmark scripts that import them.
"""
from __future__ import annotations

from flash_rt.models.imagewam.config_resolver import resolve_config
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

LIBERO_HORIZON = 64
LIBERO_STEPS = 10
LIBERO_SHIFT = 5.0

LIBERO_REAL_DIMS: dict = dict(resolve_config(ImageWAMWorkload.libero(),
                                            ImageWAMStructure.libero()).dims)

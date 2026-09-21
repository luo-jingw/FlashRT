"""Point an INT8 / INT4 benchmark at a workload.

`imagewam_thor_int8_bench.py` and `imagewam_thor_int4_bench.py` were written
for the LIBERO dims and state them as module constants (`X0`, `A0`, `TOTAL`,
`NUM_ACTION`, `VAE_IMG_H/W`, `VAE_NUM_TOKENS`). Their classes read those
constants when they are constructed, so `configure` rebinds them on the module
before the model is built. The backbone widths stay as the scripts state them
(the FLUX.2-4B backbone); what a workload changes is the sequence: the camera
grid, the text length and the action horizon.

`text_trim`: the FlashRT rows of the result tables run each prompt at its own
valid length (`x0 = valid tokens + 1`), so the INT rows must time the same
sequence; `trim=False` keeps the padded `text_max_len + 1` rows.
"""
from __future__ import annotations

from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload


def configure(module, workload: ImageWAMWorkload, *, valid_tokens: int, trim: bool = True) -> dict:
    """Rebind `module`'s sequence constants for `workload`; returns what it set."""
    layout = workload.layout(ImageWAMStructure.libero())
    x0 = valid_tokens + 1 if trim else layout.x0
    if not 0 < x0 <= layout.x0:
        raise ValueError(f"valid_tokens={valid_tokens} gives x0={x0}, outside 1..{layout.x0}")
    a0 = x0 + layout.img_len
    module.VAE_IMG_H = workload.image_h
    module.VAE_IMG_W = workload.num_views * workload.image_w
    module.VAE_NUM_TOKENS = layout.img_len
    module.MAX_ACTION_HORIZON = workload.action_horizon
    module.NUM_ACTION = workload.action_horizon
    module.X0, module.A0 = x0, a0
    module.TOTAL = a0 + workload.action_horizon
    return {"x0": x0, "a0": a0, "total": module.TOTAL, "img_len": layout.img_len,
            "num_action": workload.action_horizon, "num_steps": workload.num_steps}

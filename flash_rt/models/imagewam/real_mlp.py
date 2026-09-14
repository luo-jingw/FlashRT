"""ImageWAM/FLUX.2 real MLP (opportunities.md OPT-002 follow-up, found
while completing a full real DoubleStreamBlock forward).

Real `DoubleStreamBlock.img_mlp`/`txt_mlp` (`black-forest-labs/flux2`'s
`src/flux2/model.py`, pinned commit
`50fe5162777813d869182b139e83b10743caef15`, read directly) is:
`Linear(hidden, mlp_hidden*2, bias=False) -> SiLUActivation (chunk(2),
silu(gate)*up) -> Linear(mlp_hidden, hidden, bias=False)`. This is a
real, previously unnoticed correction to every existing
`imagewam_thor_*_bench.py` script and `pipeline_thor.py`, which all
project `hidden -> mlp_hidden` directly (9216-wide first GEMM output)
and apply plain GELU in place -- the real first GEMM is TWICE that
width (18432, `mlp_hidden*2`, "mlp_mult_factor=2" in the real source),
and the real activation is a SiLU-gated GLU chunk, not plain GELU. This
is both an accuracy AND a speed-relevant correction (the real first
GEMM does roughly 2x the FLOPs the existing benchmark scripts assumed)
-- not yet applied to those scripts or the pipeline, tracked as open in
opportunities.md.
"""
from __future__ import annotations

import flash_rt.flash_rt_kernels as fvk


def real_mlp_fp16(
    gemm,
    x: int, in_weight: int, out_weight: int,
    merged_scratch: int, gated_scratch: int, out: int,
    seq: int, hidden: int, mlp_hidden: int,
    stream: int = 0,
) -> None:
    """x: (seq, hidden) fp16 device pointer. in_weight: (mlp_hidden*2,
    hidden). out_weight: (hidden, mlp_hidden). merged_scratch: (seq,
    mlp_hidden*2). gated_scratch: (seq, mlp_hidden). Writes result to
    `out` (seq, hidden) -- caller adds the residual separately (see
    `flash_rt.models.imagewam.adaln.apply_gated_residual`).
    """
    gemm.fp16_nn(x, in_weight, merged_scratch, seq, mlp_hidden * 2, hidden, stream)
    fvk.silu_glu_merged_fp16(merged_scratch, gated_scratch, seq, mlp_hidden, stream)
    gemm.fp16_nn(gated_scratch, out_weight, out, seq, hidden, mlp_hidden, stream)

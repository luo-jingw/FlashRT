"""ImageWAM real "backbone" self-attention: QK-Norm -> RoPE -> real
masked per-head attention, combined into the order the real upstream
code actually uses.

Real order of operations, confirmed by reading `DoubleStreamBlock`/
`SingleStreamBlock` in `black-forest-labs/flux2`'s `src/flux2/model.py`
(pinned commit `50fe5162777813d869182b139e83b10743caef15`) directly:
QKNorm is applied to Q/K FIRST (`self.norm(q, k, v)` /
`self.img_attn.norm(...)`), THEN RoPE (`apply_rope(q, k, pe)`), THEN
the masked attention itself. Each of these three steps was already
implemented and independently verified against the real math elsewhere
in this project (see `tests/test_imagewam_qknorm_reuse.py`,
`tests/test_imagewam_rope_kernel.py`,
`tests/test_imagewam_backbone_ref_masked_kernel.py`) -- this module
only chains them in the correct real order, catching any interface/
ordering mismatch a per-piece test can't see (each piece alone is
correct; the composition could still be wrong if e.g. RoPE ran before
QK-Norm, or operated on the wrong buffer).

Does NOT include AdaLN modulation, LayerNorm, MLP, or residual
connections -- those are separate, not-yet-investigated parts of the
real DoubleStreamBlock/SingleStreamBlock forward, out of scope for this
combined ATTENTION-only test (see opportunities.md for what's tracked
as still open).
"""
from __future__ import annotations

import flash_rt.flash_rt_kernels as fvk


def real_backbone_attention_fp16(
    ctx,
    Q: int, K: int, V: int,
    query_norm_weight: int, key_norm_weight: int,
    rope_table: int,
    logits: int, out: int,
    total: int, NH: int, HD: int, x0: int,
    attn_scale: float,
    eps: float = 1e-6,
    stream: int = 0,
) -> None:
    """Q/K/V: (total, NH, HD) fp16 device pointers, real per-head, NOT
    broadcast. `query_norm_weight`/`key_norm_weight`: (HD,) fp16 --
    QKNorm's real learned per-head-dim scale (`query_norm.scale` /
    `key_norm.scale` in the real checkpoint). `rope_table`: (total, HD)
    fp16 interleaved cos/sin, from
    `flash_rt.models.imagewam.rope.build_backbone_rope_table`. Mutates
    Q and K in place (QK-Norm then RoPE); writes attention output to
    `out`, shape (total, NH, HD).
    """
    ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx

    # QK-Norm first (real order) -- in place, treating Q/K as (total*NH, HD)
    # rows, one per (token, head) pair (see test_imagewam_qknorm_reuse.py).
    fvk.rms_norm_fp16(Q, query_norm_weight, Q, total * NH, HD, eps, stream)
    fvk.rms_norm_fp16(K, key_norm_weight, K, total * NH, HD, eps, stream)

    # RoPE second (real order) -- in place, per (token, head) row, shared
    # cos/sin per token across all heads (see test_imagewam_rope_kernel.py).
    fvk.rope_apply_fp16_perhead(Q, rope_table, total, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(K, rope_table, total, NH, HD, stream)

    # Real masked per-head attention (see
    # test_imagewam_backbone_ref_masked_kernel.py).
    fvk.attention_qkv_fp16_backbone_ref_masked_perhead(
        ctx_cpp, Q, K, V, logits, out, total, NH, HD, x0, attn_scale, stream)

"""ImageWAM real "backbone" self-attention: QK-Norm -> RoPE -> real
per-head attention, combined into the order the real upstream code
actually uses.

Real order of operations, confirmed by reading `DoubleStreamBlock`/
`SingleStreamBlock` in `black-forest-labs/flux2`'s `src/flux2/model.py`
(pinned commit `50fe5162777813d869182b139e83b10743caef15`) directly:
QKNorm is applied to Q/K FIRST (`self.norm(q, k, v)` /
`self.img_attn.norm(...)`), THEN RoPE (`apply_rope(q, k, pe)`), THEN
the attention itself.

**No mask**: an earlier version of this module used
`attention_qkv_fp16_backbone_ref_masked_perhead` (a "txt sees all, ref
sees only itself" rule), based on `flux2/model.py`'s own
`causal_attn_fn`. Found while investigating ActionDiT's real structure
that ImageWAM's real inference path never calls that function at all --
it calls `block._prepare_qkv` directly and does its own joint attention
via `MoT._mixed_attention`, with a mask from `imagewam.py`'s
`_build_mot_attention_mask_flux2`. That function's real call sites in
`infer_action_flux2` both pass `target_len=0` (the real action-
inference path never has a separate noisy/target-image segment), which
makes the real mask rule reduce to full, unmasked visibility between
text and ref. Uses plain `attention_qkv_fp16_perhead` accordingly (see
`opportunities.md` for the full correction and
`real_double_stream_block.py`'s own docstring, which has the same
note).

Does NOT include AdaLN modulation, LayerNorm, MLP, or residual
connections -- those live in `real_double_stream_block.py`/
`real_single_stream_block.py`, which supersede this module for a full
block forward; kept as a smaller, attention-only building block.
"""
from __future__ import annotations

import flash_rt.flash_rt_kernels as fvk


def real_backbone_attention_fp16(
    ctx,
    Q: int, K: int, V: int,
    query_norm_weight: int, key_norm_weight: int,
    rope_table: int,
    logits: int, out: int,
    total: int, NH: int, HD: int,
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

    # No mask -- see module docstring for the real target_len=0 finding.
    fvk.attention_qkv_fp16_perhead(
        ctx_cpp, Q, K, V, logits, out, total, total, NH, HD, attn_scale, stream)

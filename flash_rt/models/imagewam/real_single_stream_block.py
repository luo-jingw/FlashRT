"""ImageWAM/FLUX.2 real SingleStreamBlock forward (opportunities.md
OPT-002 follow-up, after the real DoubleStreamBlock).

Real `SingleStreamBlock` (`black-forest-labs/flux2`'s
`src/flux2/model.py`, pinned commit
`50fe5162777813d869182b139e83b10743caef15`, read directly) operates on
the ALREADY-CONCATENATED `[txt | img]` combined sequence (confirmed
from `Flux2.forward`: `img = torch.cat((txt, img), dim=1)` runs once,
before the single-block loop) -- unlike `DoubleStreamBlock`, there is
only ONE stream, ONE norm, ONE modulation set here, and its QKV +
MLP-in projections are FUSED into one real `linear1` GEMM
(`Linear(hidden, 3*hidden + mlp_hidden*2)`), with attn-out-proj + MLP-
out FUSED into one real `linear2` GEMM
(`Linear(hidden + mlp_hidden, hidden)`, applied to
`cat([attn_out, mlp_act(mlp)], dim=-1)`).

**Deliberate, correctness-preserving simplification**: this module
represents `linear1` as two separate GEMMs (`qkv_weight`,
`mlp_in_weight`) and `linear2` as two separate GEMMs summed together
(`attn_out_weight`, `mlp_out_weight`) rather than one fused GEMM each.
Splitting a `Linear`'s weight matrix by output-row ranges (for
`linear1`) or input-column ranges (for `linear2`) and computing the
pieces as separate GEMMs is mathematically IDENTICAL to the fused
version -- a real checkpoint's `linear1.weight`/`linear2.weight` would
just need slicing along the appropriate axis when loaded, not
re-deriving. Chosen because `csrc/kernels/activation.cu`'s
`silu_glu_merged_fp16` kernel assumes its merged input's row stride
equals its own width (`half_dim*2`), which does not hold for a slice
carved out of a wider fused buffer; splitting into separate GEMMs
avoids needing a new strided-kernel variant for a case only this block
would use.

Real order, confirmed from `SingleStreamBlock._qkv`/`_out`:
LayerNorm -> modulate -> [qkv projection, mlp-in projection] -> QK-Norm
-> RoPE -> attention -> SiLU-gated MLP activation ->
[attn-out projection + mlp-out projection, summed] -> gated residual.

**No attention mask**: see `real_double_stream_block.py`'s own docstring
for the full correction -- ImageWAM's real inference path
(`infer_action_flux2`) always calls `_build_mot_attention_mask_flux2`
with `target_len=0`, which reduces to no masking between text and ref
tokens at all. This block operates on the whole `[txt|img]` combined
sequence, so it uses plain unmasked `attention_qkv_fp16_perhead`.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import apply_gated_residual, apply_modulation, layer_norm_no_affine_fp16

DEV = "cuda"
FP16 = torch.float16


def real_single_stream_block_forward_fp16(
    gemm, ctx,
    x: torch.Tensor,
    weights: dict,
    mod: tuple,
    rope_table: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    attn_scale: float,
    *, return_kv: bool = False,
):
    """x: (total, hidden) fp16 -- the ALREADY-CONCATENATED [txt | img]
    combined sequence. Returns the updated x (new tensor).

    `weights` keys (fp16 CUDA tensors, GEMM (K,N) convention): `qkv`,
    `mlp_in`, `attn_out`, `mlp_out`, `query_norm`, `key_norm` -- see
    module docstring for why `linear1`/`linear2` are split into two
    GEMMs each rather than one fused GEMM.

    `mod`: `(shift, scale, gate)` from
    `flash_rt.models.imagewam.adaln.modulation(..., double=False)`.

    `return_kv`: see `real_double_stream_block_forward_fp16`'s own
    docstring -- same purpose, this block's own post-QKNorm+RoPE
    per-head `(K, V)` over the whole `total` sequence. Returns
    `(x, K, V)` instead of `x` when set; default False keeps every
    existing caller unaffected.
    """
    total = x.shape[0]
    ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx
    shift, scale, gate = mod

    x_normed = layer_norm_no_affine_fp16(x)
    x_mod = apply_modulation(x_normed, shift[0].to(FP16), scale[0].to(FP16))

    qkv = torch.zeros(total, 3 * hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(x_mod.data_ptr(), weights["qkv"].data_ptr(), qkv.data_ptr(),
                 total, 3 * hidden, hidden, 0)
    Q = qkv[:, 0:hidden].reshape(total, NH, HD).contiguous()
    K = qkv[:, hidden:2 * hidden].reshape(total, NH, HD).contiguous()
    V = qkv[:, 2 * hidden:3 * hidden].reshape(total, NH, HD).contiguous()

    fvk.rms_norm_fp16(Q.data_ptr(), weights["query_norm"].data_ptr(), Q.data_ptr(),
                       total * NH, HD, 1e-6, 0)
    fvk.rms_norm_fp16(K.data_ptr(), weights["key_norm"].data_ptr(), K.data_ptr(),
                       total * NH, HD, 1e-6, 0)
    fvk.rope_apply_fp16_perhead(Q.data_ptr(), rope_table.data_ptr(), total, NH, HD, 0)
    fvk.rope_apply_fp16_perhead(K.data_ptr(), rope_table.data_ptr(), total, NH, HD, 0)

    total_pad = total + (total % 2)
    logits = torch.zeros(total * NH, total_pad, dtype=FP16, device=DEV)
    attn_out = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    # No mask: the real target_len=0 case (see module docstring) has no
    # exclusion anywhere in this already-combined sequence.
    fvk.attention_qkv_fp16_perhead(
        ctx_cpp, Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        logits.data_ptr(), attn_out.data_ptr(), total, total, NH, HD, attn_scale, 0)
    attn_out_flat = attn_out.reshape(total, hidden)

    mlp_merged = torch.zeros(total, mlp_hidden * 2, dtype=FP16, device=DEV)
    gemm.fp16_nn(x_mod.data_ptr(), weights["mlp_in"].data_ptr(), mlp_merged.data_ptr(),
                 total, mlp_hidden * 2, hidden, 0)
    mlp_gated = torch.zeros(total, mlp_hidden, dtype=FP16, device=DEV)
    fvk.silu_glu_merged_fp16(mlp_merged.data_ptr(), mlp_gated.data_ptr(), total, mlp_hidden, 0)

    from_attn = torch.zeros(total, hidden, dtype=FP16, device=DEV)
    from_mlp = torch.zeros(total, hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(attn_out_flat.data_ptr(), weights["attn_out"].data_ptr(), from_attn.data_ptr(),
                 total, hidden, hidden, 0)
    gemm.fp16_nn(mlp_gated.data_ptr(), weights["mlp_out"].data_ptr(), from_mlp.data_ptr(),
                 total, hidden, mlp_hidden, 0)
    torch.cuda.synchronize()

    output = (from_attn.float() + from_mlp.float()).to(FP16)
    x_out = apply_gated_residual(x.float(), output.float(), gate[0].float()).to(FP16)
    if return_kv:
        return x_out, K, V
    return x_out

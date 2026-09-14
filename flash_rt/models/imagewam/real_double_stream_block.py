"""ImageWAM/FLUX.2 real DoubleStreamBlock forward -- the final piece
combining every real-math correction found this round (per-head K/V,
RoPE, QK-Norm, AdaLN modulation, real LayerNorm, real SiLU-gated MLP)
into one full single-layer block forward.

**Mask correction**: an earlier version of this module used a real
"txt sees all, ref sees only itself" mask, based on `flux2/model.py`'s
own `causal_attn_fn`. Found while investigating ActionDiT's real
structure that this is the WRONG source function -- ImageWAM's own
inference path never calls `causal_attn_fn`/`forward_kv_extract` at
all; it calls `block._prepare_qkv` directly and does its OWN joint
attention via `MoT._mixed_attention`, with a mask built by
`imagewam.py`'s `_build_mot_attention_mask_flux2`. Read that function
AND its real call sites in `infer_action_flux2` directly: both mask
constructions there pass `target_len=0` (the real action-inference path
never has a separate noisy/target-image segment -- only text, a
reference image, and, later, action tokens exist). With
`target_len=0`, `_build_mot_attention_mask_flux2`'s own rule reduces to
NO masking at all between text and ref (`mask[text, text:ref]=True` AND
`mask[ref, text:ref]=True` -- both directions, full visibility) --
confirmed directly from the real source, not re-derived. This module
now uses plain unmasked `attention_qkv_fp16_perhead` accordingly.

Real order, confirmed by reading `DoubleStreamBlock._prepare_qkv`/
`_apply_residuals` in `black-forest-labs/flux2`'s `src/flux2/model.py`
(pinned commit `50fe5162777813d869182b139e83b10743caef15`) directly:

    txt_mod = (1+txt_mod1_scale)*LayerNorm(txt) + txt_mod1_shift
    img_mod = (1+img_mod1_scale)*LayerNorm(img) + img_mod1_shift
    txt_q,txt_k,txt_v = split_heads(txt_attn.qkv(txt_mod))
    img_q,img_k,img_v = split_heads(img_attn.qkv(img_mod))
    txt_q,txt_k = QKNorm(txt_q,txt_k)      # txt_attn.norm
    img_q,img_k = QKNorm(img_q,img_k)      # img_attn.norm
    q,k,v = cat([txt_*, img_*])            # [txt | ref-image] combined
    q,k = RoPE(q,k)
    attn = attention(q,k,v)                # NO mask (target_len=0 case)
    txt_attn_out, img_attn_out = split(attn)
    txt = txt + txt_mod1_gate * txt_attn.proj(txt_attn_out)
    img = img + img_mod1_gate * img_attn.proj(img_attn_out)
    txt = txt + txt_mod2_gate * txt_mlp((1+txt_mod2_scale)*LayerNorm(txt) + txt_mod2_shift)
    img = img + img_mod2_gate * img_mlp((1+img_mod2_scale)*LayerNorm(img) + img_mod2_shift)

This module is a verification primitive, not yet wired into
`pipeline_thor.py` (see opportunities.md for what remains before that:
real checkpoint access, `_imagewam_thor_spec.py` weight-shape changes --
not yet done here).
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import apply_gated_residual, apply_modulation, layer_norm_no_affine_fp16
from flash_rt.models.imagewam.real_mlp import real_mlp_fp16

DEV = "cuda"
FP16 = torch.float16


def real_double_stream_block_forward_fp16(
    gemm, ctx,
    txt: torch.Tensor, img: torch.Tensor,
    weights: dict,
    mod_txt: tuple, mod_img: tuple,
    rope_table: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    attn_scale: float,
    *, return_kv: bool = False,
):
    """txt: (x0, hidden) fp16. img: (img_len, hidden) fp16. Both updated
    and returned (new tensors, not in-place, for test clarity).

    `weights` keys (all fp16 CUDA tensors, GEMM (K,N) convention --
    transpose from a real nn.Linear's own (out,in) weight if loading
    real checkpoint tensors): `txt_qkv`, `txt_proj`, `txt_mlp_in`,
    `txt_mlp_out`, `txt_query_norm`, `txt_key_norm`, and the `img_*`
    analogs of all six.

    `mod_txt`/`mod_img`: `((shift1,scale1,gate1), (shift2,scale2,gate2))`
    from `flash_rt.models.imagewam.adaln.modulation(..., double=True)`.

    `return_kv`: when True, also returns this layer's own post-QKNorm+
    RoPE per-head `(K, V)` (each `(x0+img_len, NH, HD)`, in the same
    `[txt | img]` row order as this block's own combined attention
    call) -- needed by `pipeline_real.imagewam_full_forward_real` to
    build the per-layer frozen K/V cache the action expert's joint
    attention reads later (see `real_action_expert.py`'s own
    `cached_k`/`cached_v` parameters). Returns `(txt, img, K, V)`
    instead of `(txt, img)` when set; default False keeps every
    existing caller (`imagewam_prefill_real`, this module's own tests)
    unaffected.
    """
    x0, img_len = txt.shape[0], img.shape[0]
    total = x0 + img_len
    ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx

    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = mod_txt
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = mod_img

    txt_normed1 = layer_norm_no_affine_fp16(txt)
    img_normed1 = layer_norm_no_affine_fp16(img)
    txt_mod1 = apply_modulation(txt_normed1, txt_shift1[0].to(FP16), txt_scale1[0].to(FP16))
    img_mod1 = apply_modulation(img_normed1, img_shift1[0].to(FP16), img_scale1[0].to(FP16))

    # Combined Q/K/V buffer, [txt | img] order (matches the real
    # concatenation order and this project's own real-mask convention).
    Q = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    K = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    V = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    txt_qkv = torch.zeros(x0, 3 * hidden, dtype=FP16, device=DEV)
    img_qkv = torch.zeros(img_len, 3 * hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(txt_mod1.data_ptr(), weights["txt_qkv"].data_ptr(), txt_qkv.data_ptr(),
                 x0, 3 * hidden, hidden, 0)
    gemm.fp16_nn(img_mod1.data_ptr(), weights["img_qkv"].data_ptr(), img_qkv.data_ptr(),
                 img_len, 3 * hidden, hidden, 0)

    Q[:x0] = txt_qkv[:, 0:hidden].reshape(x0, NH, HD)
    K[:x0] = txt_qkv[:, hidden:2 * hidden].reshape(x0, NH, HD)
    V[:x0] = txt_qkv[:, 2 * hidden:3 * hidden].reshape(x0, NH, HD)
    Q[x0:] = img_qkv[:, 0:hidden].reshape(img_len, NH, HD)
    K[x0:] = img_qkv[:, hidden:2 * hidden].reshape(img_len, NH, HD)
    V[x0:] = img_qkv[:, 2 * hidden:3 * hidden].reshape(img_len, NH, HD)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # QK-Norm: txt rows and img rows use DIFFERENT learned scales
    # (txt_attn.norm vs img_attn.norm) -- applied to each region
    # separately, in place.
    fvk.rms_norm_fp16(Q[:x0].data_ptr(), weights["txt_query_norm"].data_ptr(), Q[:x0].data_ptr(),
                       x0 * NH, HD, 1e-6, 0)
    fvk.rms_norm_fp16(K[:x0].data_ptr(), weights["txt_key_norm"].data_ptr(), K[:x0].data_ptr(),
                       x0 * NH, HD, 1e-6, 0)
    fvk.rms_norm_fp16(Q[x0:].data_ptr(), weights["img_query_norm"].data_ptr(), Q[x0:].data_ptr(),
                       img_len * NH, HD, 1e-6, 0)
    fvk.rms_norm_fp16(K[x0:].data_ptr(), weights["img_key_norm"].data_ptr(), K[x0:].data_ptr(),
                       img_len * NH, HD, 1e-6, 0)

    # RoPE: shared position table over the whole combined sequence.
    fvk.rope_apply_fp16_perhead(Q.data_ptr(), rope_table.data_ptr(), total, NH, HD, 0)
    fvk.rope_apply_fp16_perhead(K.data_ptr(), rope_table.data_ptr(), total, NH, HD, 0)

    total_pad = total + (total % 2)
    logits = torch.zeros(total * NH, total_pad, dtype=FP16, device=DEV)
    attn_out = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    # No mask: the real target_len=0 case (see module docstring) has no
    # exclusion between text and ref -- plain full self-attention.
    fvk.attention_qkv_fp16_perhead(
        ctx_cpp, Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        logits.data_ptr(), attn_out.data_ptr(), total, total, NH, HD, attn_scale, 0)

    attn_out_flat = attn_out.reshape(total, hidden)
    txt_attn_out = attn_out_flat[:x0].contiguous()
    img_attn_out = attn_out_flat[x0:].contiguous()

    txt_proj_out = torch.zeros(x0, hidden, dtype=FP16, device=DEV)
    img_proj_out = torch.zeros(img_len, hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(txt_attn_out.data_ptr(), weights["txt_proj"].data_ptr(), txt_proj_out.data_ptr(),
                 x0, hidden, hidden, 0)
    gemm.fp16_nn(img_attn_out.data_ptr(), weights["img_proj"].data_ptr(), img_proj_out.data_ptr(),
                 img_len, hidden, hidden, 0)

    txt = apply_gated_residual(txt.float(), txt_proj_out.float(), txt_gate1[0].float()).to(FP16)
    img = apply_gated_residual(img.float(), img_proj_out.float(), img_gate1[0].float()).to(FP16)

    txt_normed2 = layer_norm_no_affine_fp16(txt)
    img_normed2 = layer_norm_no_affine_fp16(img)
    txt_mod2 = apply_modulation(txt_normed2, txt_shift2[0].to(FP16), txt_scale2[0].to(FP16))
    img_mod2 = apply_modulation(img_normed2, img_shift2[0].to(FP16), img_scale2[0].to(FP16))

    txt_mlp_merged = torch.zeros(x0, mlp_hidden * 2, dtype=FP16, device=DEV)
    txt_mlp_gated = torch.zeros(x0, mlp_hidden, dtype=FP16, device=DEV)
    txt_mlp_out = torch.zeros(x0, hidden, dtype=FP16, device=DEV)
    real_mlp_fp16(gemm, txt_mod2.data_ptr(), weights["txt_mlp_in"].data_ptr(),
                  weights["txt_mlp_out"].data_ptr(), txt_mlp_merged.data_ptr(),
                  txt_mlp_gated.data_ptr(), txt_mlp_out.data_ptr(), x0, hidden, mlp_hidden, 0)

    img_mlp_merged = torch.zeros(img_len, mlp_hidden * 2, dtype=FP16, device=DEV)
    img_mlp_gated = torch.zeros(img_len, mlp_hidden, dtype=FP16, device=DEV)
    img_mlp_out = torch.zeros(img_len, hidden, dtype=FP16, device=DEV)
    real_mlp_fp16(gemm, img_mod2.data_ptr(), weights["img_mlp_in"].data_ptr(),
                  weights["img_mlp_out"].data_ptr(), img_mlp_merged.data_ptr(),
                  img_mlp_gated.data_ptr(), img_mlp_out.data_ptr(), img_len, hidden, mlp_hidden, 0)

    torch.cuda.synchronize()

    txt = apply_gated_residual(txt.float(), txt_mlp_out.float(), txt_gate2[0].float()).to(FP16)
    img = apply_gated_residual(img.float(), img_mlp_out.float(), img_gate2[0].float()).to(FP16)

    if return_kv:
        return txt, img, K, V
    return txt, img

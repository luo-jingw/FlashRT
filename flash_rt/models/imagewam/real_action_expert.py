"""ImageWAM real ActionDiT (action expert) forward -- the "mot" site,
completing OPT-002's real-math coverage alongside the backbone.

Real structure, confirmed by reading `ActionDiTFlux2`/`SlimFlux2DoubleBlock`/
`SlimFlux2SingleBlock` (ImageWAM's own source, `action_dit_flux2.py` --
NOT `flux2/model.py`, a separate, slimmer model reusing FLUX.2's own
`QKNorm`/`SiLUActivation` building blocks) and the real joint-attention
orchestration in `mot.py`'s `forward_flux2_action_with_video_cache`:

- ActionDiT's own double/single blocks are structurally identical to
  the backbone's, but IMG-ONLY -- there is no separate txt branch here
  at all (`SlimFlux2DoubleBlock` only has `img_norm1/img_attn/img_norm2/
  img_mlp`, no `txt_*` equivalent), since ActionDiT operates on action
  tokens alone.
- The actual joint attention is orchestrated EXTERNALLY by `MoT`, not
  internal to the block: `block.prepare_qkv(action, action_pe, mod)`
  returns RAW Q/K/V for the action tokens only; the caller then
  concatenates the FROZEN backbone K/V cache (from that same layer's
  own backbone forward) with action's own fresh K/V
  (`k_cat = cat([cached_k, action_k])`), runs ONE attention call with Q
  = action rows only and K/V = the full concatenated sequence, and only
  THEN calls `block.apply_post(mixed_attn_out, state)` to finish the
  residual/MLP wiring.
- **No mask** (opportunities.md's real-mask correction): ImageWAM's own
  `_build_mot_attention_mask_flux2`, with the real `target_len=0` this
  project's deployment target always uses, gives action full visibility
  over `[text, ref, action]` -- not an exclusion of the image region,
  which is what this project's pre-existing "mot_joint"/
  "mot_joint_action" kernels (OPT-003) assumed. This module uses plain
  unmasked `attention_qkv_fp16_perhead` with `S=num_action`,
  `S_kv=total` (the same kernel already used for the corrected
  backbone, just with S != S_kv here since action's own query count is
  smaller than the full K/V it attends to).
- ActionDiT's own RoPE uses a DIFFERENT position convention from the
  backbone's (`build_action_ids`: axis0=2.0 constant "type marker",
  axis1=running index) but the SAME `pe_embedder` config (axes_dim,
  theta) -- confirmed from `mot.py`'s own
  `action_pe = video_expert.transformer.pe_embedder(action_ids)` call,
  reusing the backbone's embedder object directly. Only action's own
  fresh Q/K get this RoPE applied here; the cached backbone K was
  already rotated (with ITS OWN position ids) during the backbone's own
  forward and never needs re-rotating.
- **`attn_dim != hidden` here, unlike the backbone**: real
  `SlimFlux2SelfAttention` (`action_dit_flux2.py`) has
  `attn_dim = num_heads * attn_head_dim` (3072, matching this project's
  own `ACTION_ATTN_WIDTH`) while ActionDiT's residual-stream width is
  its own, SMALLER `hidden_dim` (1024, `ACTION_HIDDEN_DIM`) --
  `qkv: Linear(hidden_dim, 3*attn_dim)`, `proj: Linear(attn_dim,
  hidden_dim)`. The backbone's own blocks don't have this distinction
  (`num_heads*head_dim == hidden_size` there, confirmed from
  `DoubleStreamBlock.__init__`'s own assert). Every function below
  takes `attn_dim` and `hidden` as separate parameters accordingly.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import apply_gated_residual, apply_modulation, layer_norm_no_affine_fp16
from flash_rt.models.imagewam.real_mlp import real_mlp_fp16

DEV = "cuda"
FP16 = torch.float16


def _prepare_action_qkv(gemm, action, qkv_weight, query_norm, key_norm, action_rope_table,
                         NH, HD, hidden, attn_dim):
    num_action = action.shape[0]
    qkv = torch.zeros(num_action, 3 * attn_dim, dtype=FP16, device=DEV)
    gemm.fp16_nn(action.data_ptr(), qkv_weight.data_ptr(), qkv.data_ptr(),
                 num_action, 3 * attn_dim, hidden, 0)
    Q = qkv[:, 0:attn_dim].reshape(num_action, NH, HD).contiguous()
    K = qkv[:, attn_dim:2 * attn_dim].reshape(num_action, NH, HD).contiguous()
    V = qkv[:, 2 * attn_dim:3 * attn_dim].reshape(num_action, NH, HD).contiguous()

    fvk.rms_norm_fp16(Q.data_ptr(), query_norm.data_ptr(), Q.data_ptr(), num_action * NH, HD, 1e-6, 0)
    fvk.rms_norm_fp16(K.data_ptr(), key_norm.data_ptr(), K.data_ptr(), num_action * NH, HD, 1e-6, 0)
    fvk.rope_apply_fp16_perhead(Q.data_ptr(), action_rope_table.data_ptr(), num_action, NH, HD, 0)
    fvk.rope_apply_fp16_perhead(K.data_ptr(), action_rope_table.data_ptr(), num_action, NH, HD, 0)
    return Q, K, V


def _joint_attention(Q, K, V, cached_k, cached_v, NH, HD, attn_scale):
    num_action = Q.shape[0]
    K_cat = torch.cat([cached_k, K], dim=0).contiguous()
    V_cat = torch.cat([cached_v, V], dim=0).contiguous()
    total = K_cat.shape[0]

    ctx = fvk.FvkContext()
    total_pad = total + (total % 2)
    logits = torch.zeros(num_action * NH, total_pad, dtype=FP16, device=DEV)
    attn_out = torch.zeros(num_action, NH, HD, dtype=FP16, device=DEV)
    # No mask (see module docstring): action sees the whole concatenated
    # [cached backbone K/V | action's own K/V] sequence.
    fvk.attention_qkv_fp16_perhead(
        ctx, Q.data_ptr(), K_cat.data_ptr(), V_cat.data_ptr(),
        logits.data_ptr(), attn_out.data_ptr(), num_action, total, NH, HD, attn_scale, 0)
    return attn_out.reshape(num_action, NH * HD)


def real_action_double_block_forward_fp16(
    gemm,
    action: torch.Tensor,
    weights: dict,
    mod: tuple,
    action_rope_table: torch.Tensor,
    cached_k: torch.Tensor, cached_v: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    attn_scale: float,
) -> torch.Tensor:
    """Real `SlimFlux2DoubleBlock.prepare_qkv`/`apply_post`, IMG-ONLY
    (no txt branch -- see module docstring). `weights` keys: `qkv`,
    `proj`, `mlp_in`, `mlp_out`, `query_norm`, `key_norm` (fp16 CUDA
    tensors, GEMM (K,N) convention). `mod`: `((shift1,scale1,gate1),
    (shift2,scale2,gate2))` from `modulation(..., double=True)`.
    `cached_k`/`cached_v`: this layer's own frozen backbone K/V cache,
    (backbone_total, NH, HD). `attn_dim = NH*HD`, independent of
    `hidden` -- see module docstring.
    """
    attn_dim = NH * HD
    (shift1, scale1, gate1), (shift2, scale2, gate2) = mod

    x_normed1 = layer_norm_no_affine_fp16(action)
    x_mod1 = apply_modulation(x_normed1, shift1[0].to(FP16), scale1[0].to(FP16))

    Q, K, V = _prepare_action_qkv(gemm, x_mod1, weights["qkv"], weights["query_norm"],
                                   weights["key_norm"], action_rope_table, NH, HD, hidden, attn_dim)
    mixed = _joint_attention(Q, K, V, cached_k, cached_v, NH, HD, attn_scale)

    proj_out = torch.zeros(action.shape[0], hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(mixed.data_ptr(), weights["proj"].data_ptr(), proj_out.data_ptr(),
                 action.shape[0], hidden, attn_dim, 0)
    action = apply_gated_residual(action.float(), proj_out.float(), gate1[0].float()).to(FP16)

    x_normed2 = layer_norm_no_affine_fp16(action)
    x_mod2 = apply_modulation(x_normed2, shift2[0].to(FP16), scale2[0].to(FP16))
    num_action = action.shape[0]
    mlp_merged = torch.zeros(num_action, mlp_hidden * 2, dtype=FP16, device=DEV)
    mlp_gated = torch.zeros(num_action, mlp_hidden, dtype=FP16, device=DEV)
    mlp_out = torch.zeros(num_action, hidden, dtype=FP16, device=DEV)
    real_mlp_fp16(gemm, x_mod2.data_ptr(), weights["mlp_in"].data_ptr(), weights["mlp_out"].data_ptr(),
                  mlp_merged.data_ptr(), mlp_gated.data_ptr(), mlp_out.data_ptr(),
                  num_action, hidden, mlp_hidden, 0)
    torch.cuda.synchronize()

    return apply_gated_residual(action.float(), mlp_out.float(), gate2[0].float()).to(FP16)


def real_action_single_block_forward_fp16(
    gemm,
    action: torch.Tensor,
    weights: dict,
    mod: tuple,
    action_rope_table: torch.Tensor,
    cached_k: torch.Tensor, cached_v: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    attn_scale: float,
) -> torch.Tensor:
    """Real `SlimFlux2SingleBlock.prepare_qkv`/`apply_post`. `weights`
    keys: `qkv`, `mlp_in`, `attn_out`, `mlp_out`, `query_norm`,
    `key_norm` (same split-linear1/linear2 simplification as
    `real_single_stream_block.py`, see that module's own docstring).
    `mod`: `(shift, scale, gate)` from `modulation(..., double=False)`.
    `attn_dim = NH*HD`, independent of `hidden` -- see module docstring.
    """
    attn_dim = NH * HD
    shift, scale, gate = mod
    num_action = action.shape[0]

    x_normed = layer_norm_no_affine_fp16(action)
    x_mod = apply_modulation(x_normed, shift[0].to(FP16), scale[0].to(FP16))

    Q, K, V = _prepare_action_qkv(gemm, x_mod, weights["qkv"], weights["query_norm"],
                                   weights["key_norm"], action_rope_table, NH, HD, hidden, attn_dim)
    mixed = _joint_attention(Q, K, V, cached_k, cached_v, NH, HD, attn_scale)

    mlp_merged = torch.zeros(num_action, mlp_hidden * 2, dtype=FP16, device=DEV)
    gemm.fp16_nn(x_mod.data_ptr(), weights["mlp_in"].data_ptr(), mlp_merged.data_ptr(),
                 num_action, mlp_hidden * 2, hidden, 0)
    mlp_gated = torch.zeros(num_action, mlp_hidden, dtype=FP16, device=DEV)
    fvk.silu_glu_merged_fp16(mlp_merged.data_ptr(), mlp_gated.data_ptr(), num_action, mlp_hidden, 0)

    from_attn = torch.zeros(num_action, hidden, dtype=FP16, device=DEV)
    from_mlp = torch.zeros(num_action, hidden, dtype=FP16, device=DEV)
    gemm.fp16_nn(mixed.data_ptr(), weights["attn_out"].data_ptr(), from_attn.data_ptr(),
                 num_action, hidden, attn_dim, 0)
    gemm.fp16_nn(mlp_gated.data_ptr(), weights["mlp_out"].data_ptr(), from_mlp.data_ptr(),
                 num_action, hidden, mlp_hidden, 0)
    torch.cuda.synchronize()

    output = (from_attn.float() + from_mlp.float()).to(FP16)
    return apply_gated_residual(action.float(), output.float(), gate[0].float()).to(FP16)

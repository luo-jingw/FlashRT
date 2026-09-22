"""ImageWAM `pipeline_thor.py`'s real-math rewrite (2026-09-14,
opportunities.md OPT-002 / PROJECT.md "Confirmed end goal") -- pointer-
based, buffer-reusing, `ImageWAMAttnBackend`-dispatched layer helpers
must match the already-verified TENSOR-level reference
(`real_double_stream_block_forward_fp16` / `real_single_stream_block_forward_fp16`
/ `real_action_double_block_forward_fp16` / `real_action_single_block_forward_fp16`,
themselves independently verified against the real trained checkpoint
on Thor).

This is the highest-risk part of the rewrite: everything here is a
hand-translation from tensor-returning calls to pointer-offset/buffer-
reuse code (fused QKV split into 3 separate GEMMs, AdaLN modulation
applied via zero-copy tensor VIEWS instead of owned tensors, attention
dispatched through `ImageWAMAttnBackend` instead of a direct kernel
call) -- a pointer-arithmetic bug here would not show up as a crash,
only as silently wrong numbers. Each layer type is checked by building
IDENTICAL weights/inputs for both call conventions (a fused QKV matrix
is split by COLUMN range into separate q/k/v matrices -- mathematically
identical, see `_imagewam_thor_spec.py`'s own docstring) and comparing
outputs by cosine similarity.
"""
import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
from flash_rt.models.imagewam.pipeline_real import compute_action_modulation, compute_shared_modulation
from flash_rt.models.imagewam.pipeline_thor import (
    AdaLNTarget, _action_double_layer, _action_single_layer, _awq_target,
    _double_stream_layer, _single_stream_layer)
from flash_rt.models.imagewam.quant_linear import Fp16Linear, Nvfp4Linear
from flash_rt.models.imagewam.real_action_expert import real_action_double_block_forward_fp16, real_action_single_block_forward_fp16
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16
from flash_rt.models.imagewam.real_single_stream_block import real_single_stream_block_forward_fp16
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
F32 = torch.float32
AXES_DIM = (32, 32, 32, 32)
THETA = 2000

_keepalive = []


def _own(t):
    _keepalive.append(t)
    return t


def _fused_qkv(hidden, device):
    """One random fused (hidden, 3*hidden) QKV weight (GEMM (K,N)
    convention) -- this IS the real fused `{prefix}_qkv.weight` the
    pointer path now consumes directly (OPT-004 step 5, QKV fusion);
    `q`/`k`/`v` (contiguous column-slice copies) are still returned
    for the tensor-level reference (`real_double_stream_block_forward_fp16`
    etc.) which builds its own `torch.cat([q,k,v],dim=1)`."""
    fused = _own((torch.randn(hidden, 3 * hidden, dtype=torch.float32, device=device) * 0.02).to(FP16))
    q = fused[:, 0:hidden].contiguous()
    k = fused[:, hidden:2 * hidden].contiguous()
    v = fused[:, 2 * hidden:3 * hidden].contiguous()
    return q, k, v, fused


def _lin(n, k, device):
    return _own((torch.randn(n, k, dtype=torch.float32, device=device) * 0.02).to(FP16)).t().contiguous()


def _norm_scale(HD, device):
    return _own((torch.randn(HD, dtype=torch.float32, device=device).abs() + 0.5).to(FP16))


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_double_stream_layer_matches_real_reference():
    x0, img_len, NH, HD, mlp_hidden = 3, 5, 2, 128, 192
    hidden = NH * HD
    a0 = x0 + img_len
    joint_attention_dim = 40
    torch.manual_seed(0)
    scale = 1.0 / (HD ** 0.5)

    gemm = fvk.GemmRunner()

    context = _own(torch.randn(x0, joint_attention_dim, dtype=FP16, device=DEV) * 0.1)
    txt_in_w = _lin(hidden, joint_attention_dim, DEV)
    img_in_w = _lin(hidden, HD, DEV)
    img_raw = _own(torch.randn(img_len, HD, dtype=FP16, device=DEV) * 0.1)

    ref_w, ptr_w = {}, {
        ("backbone", "double", 0, "txt_in.weight"):
            Fp16Linear(gemm, txt_in_w.data_ptr(), hidden, joint_attention_dim),
        ("backbone", "double", 0, "img_in.weight"):
            Fp16Linear(gemm, img_in_w.data_ptr(), hidden, HD),
    }
    for side in ("txt", "img"):
        q, k, v, fused = _fused_qkv(hidden, DEV)
        ref_w[f"{side}_qkv"] = torch.cat([q, k, v], dim=1)
        ptr_w[("backbone", "double", 0, f"{side}_qkv.weight")] = Fp16Linear(gemm, fused.data_ptr(), 3 * hidden, hidden)
        proj = _lin(hidden, hidden, DEV)
        ref_w[f"{side}_proj"] = proj
        ptr_w[("backbone", "double", 0, f"{side}_proj.weight")] = Fp16Linear(gemm, proj.data_ptr(), hidden, hidden)
        mlp_in = _lin(mlp_hidden * 2, hidden, DEV)
        ref_w[f"{side}_mlp_in"] = mlp_in
        ptr_w[("backbone", "double", 0, f"{side}_mlp0.weight")] = Fp16Linear(gemm, mlp_in.data_ptr(), mlp_hidden * 2, hidden)
        mlp_out = _lin(hidden, mlp_hidden, DEV)
        ref_w[f"{side}_mlp_out"] = mlp_out
        ptr_w[("backbone", "double", 0, f"{side}_mlp2.weight")] = Fp16Linear(gemm, mlp_out.data_ptr(), hidden, mlp_hidden)
        qn = _norm_scale(HD, DEV)
        ref_w[f"{side}_query_norm"] = qn
        ptr_w[("backbone", "double", 0, f"{side}_query_norm")] = qn.data_ptr()
        kn = _norm_scale(HD, DEV)
        ref_w[f"{side}_key_norm"] = kn
        ptr_w[("backbone", "double", 0, f"{side}_key_norm")] = kn.data_ptr()

    mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    timestep = torch.zeros(1, device=DEV)
    mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
    table = build_backbone_rope_table(x0, img_len, 1, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    ctx = fvk.FvkContext()

    # Derive the exact same txt/img inputs both paths will use, via the
    # SAME gemm.fp16_nn calls the pointer path itself makes (OPT-001/
    # OPT-008: img now needs its own img_in projection first, mirroring
    # txt_in's own already-established pattern here).
    txt_input = _own(torch.zeros(x0, hidden, dtype=FP16, device=DEV))
    gemm.fp16_nn(context.data_ptr(), txt_in_w.data_ptr(), txt_input.data_ptr(), x0, hidden, joint_attention_dim, 0)
    img = _own(torch.zeros(img_len, hidden, dtype=FP16, device=DEV))
    gemm.fp16_nn(img_raw.data_ptr(), img_in_w.data_ptr(), img.data_ptr(), img_len, hidden, HD, 0)

    txt_ref, img_ref = real_double_stream_block_forward_fp16(
        gemm, ctx, txt_input.clone(), img.clone(), ref_w, mod_txt, mod_img, table, NH, HD, hidden, mlp_hidden, scale)

    # Pointer-based path: one "backbone" site, one double layer.
    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=a0 + 1,
                                         num_layers=1, num_heads=NH, head_dim=HD)
    K_cache = _own(torch.zeros(1, a0, hidden, dtype=FP16, device=DEV))
    V_cache = _own(torch.zeros(1, a0, hidden, dtype=FP16, device=DEV))
    Q_O = _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV))
    logits = _own(torch.zeros(a0 * NH, a0 + (a0 % 2), dtype=FP16, device=DEV))
    attn = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                        "logits": logits.data_ptr(), "scale": scale},
        mot_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                   "logits": logits.data_ptr(), "scale": scale, "layer_stride": K_cache[0].numel() * 2},
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    # BF16, not FP16 -- `_double_stream_layer`'s AdaLN/gate_res calls
    # now unconditionally read/write `combined` as BF16 (opportunities.md
    # OPT-001 "FP16 residual overflow" fix); FP16 here would silently
    # reinterpret the wrong bit pattern, not just lose precision.
    combined = _own(torch.zeros(a0, hidden, dtype=BF16, device=DEV))
    # `_double_stream_layer` no longer projects txt_in/img_in itself
    # (bug fix, 2026-09-15, opportunities.md OPT-001: it used to redo
    # this every layer, discarding the previous layer's output -- a
    # single-layer test like this one could never have caught that,
    # since there IS no previous layer here). `imagewam_prefill` now
    # does this ONCE before its layer loop; this test does the
    # equivalent explicitly, matching `txt_input`/`img`'s own
    # construction above exactly so both paths see the same input --
    # via BF16 versions of the same weights/inputs (`gemm.bf16_nn`
    # needs A/B/D all the same dtype), mirroring `Bf16OutLinear`.
    context_bf16 = _own(context.to(BF16))
    txt_in_w_bf16 = _own(txt_in_w.to(BF16))
    img_raw_bf16 = _own(img_raw.to(BF16))
    img_in_w_bf16 = _own(img_in_w.to(BF16))
    gemm.bf16_nn(context_bf16.data_ptr(), txt_in_w_bf16.data_ptr(), combined.data_ptr(),
                 x0, hidden, joint_attention_dim, 0)
    gemm.bf16_nn(img_raw_bf16.data_ptr(), img_in_w_bf16.data_ptr(),
                 combined.data_ptr() + x0 * hidden * 2, img_len, hidden, HD, 0)
    bufs = {
        "context": context.data_ptr(),
        "img_raw": img_raw.data_ptr(),
        "backbone_hidden": combined.data_ptr(),
        "modded_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
        "txt_qkv_merged": _own(torch.zeros(x0, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
        "img_qkv_merged": _own(torch.zeros(img_len, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
        "txt_mlp_merged": _own(torch.zeros(x0, mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
        "txt_mlp_gated": _own(torch.zeros(x0, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "img_mlp_merged": _own(torch.zeros(img_len, mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
        "img_mlp_gated": _own(torch.zeros(img_len, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "proj_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden,
                joint_attention_dim=joint_attention_dim, x0=x0, a0=a0)

    _double_stream_layer(ctx, fvk, gemm, bufs, ptr_w, dims, 0, 0, attn, mod_txt, mod_img, table.data_ptr())

    ref_combined = torch.cat([txt_ref, img_ref], dim=0)
    cos = _cosine(combined, ref_combined)
    print(f"double-stream layer: pointer path vs real reference cosine={cos:.6f}")
    assert cos > 0.999


def test_single_stream_layer_matches_real_reference():
    NH, HD, mlp_hidden = 2, 128, 192
    hidden = NH * HD
    total = 8
    torch.manual_seed(1)
    scale = 1.0 / (HD ** 0.5)

    gemm = fvk.GemmRunner()

    q, k, v, fused = _fused_qkv(hidden, DEV)
    ref_w = {
        "qkv": torch.cat([q, k, v], dim=1),
        "attn_out": _lin(hidden, hidden, DEV),
        "mlp_in": _lin(mlp_hidden * 2, hidden, DEV),
        "mlp_out": _lin(hidden, mlp_hidden, DEV),
        "query_norm": _norm_scale(HD, DEV),
        "key_norm": _norm_scale(HD, DEV),
    }
    ptr_w = {
        ("backbone", "single", 0, "qkv.weight"): Fp16Linear(gemm, fused.data_ptr(), 3 * hidden, hidden),
        ("backbone", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, ref_w["attn_out"].data_ptr(), hidden, hidden),
        ("backbone", "single", 0, "mlp_in.weight"): Fp16Linear(gemm, ref_w["mlp_in"].data_ptr(), mlp_hidden * 2, hidden),
        ("backbone", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, ref_w["mlp_out"].data_ptr(), hidden, mlp_hidden),
        ("backbone", "single", 0, "query_norm"): ref_w["query_norm"].data_ptr(),
        ("backbone", "single", 0, "key_norm"): ref_w["key_norm"].data_ptr(),
    }

    mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    timestep = torch.zeros(1, device=DEV)
    _, _, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
    table = build_backbone_rope_table(3, 5, 1, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    ctx = fvk.FvkContext()

    x = _own(torch.randn(total, hidden, dtype=FP16, device=DEV) * 0.1)
    x_ref = real_single_stream_block_forward_fp16(
        gemm, ctx, x.clone(), ref_w, mod_single, table, NH, HD, hidden, mlp_hidden, scale)

    spec = make_imagewam_attention_spec(max_prefix_seq=total, max_total_seq=total + 1,
                                         num_layers=1, num_heads=NH, head_dim=HD)
    K_cache = _own(torch.zeros(1, total, hidden, dtype=FP16, device=DEV))
    V_cache = _own(torch.zeros(1, total, hidden, dtype=FP16, device=DEV))
    Q_O = _own(torch.zeros(total, hidden, dtype=FP16, device=DEV))
    logits = _own(torch.zeros(total * NH, total + (total % 2), dtype=FP16, device=DEV))
    attn = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                        "logits": logits.data_ptr(), "scale": scale},
        mot_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                   "logits": logits.data_ptr(), "scale": scale, "layer_stride": K_cache[0].numel() * 2},
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    # BF16, not FP16 -- `_single_stream_layer`'s AdaLN/gate_res calls
    # now unconditionally read/write `combined` as BF16 (opportunities.md
    # OPT-001 "FP16 residual overflow" fix); `x_ref` (the separate
    # PyTorch-level reference) stays FP16, unaffected.
    combined = _own(x.clone().to(BF16))
    bufs = {
        "backbone_hidden": combined.data_ptr(),
        "modded_scratch": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
        "single_qkv_merged": _own(torch.zeros(total, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
        "single_mlp_merged": _own(torch.zeros(total, mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
        "single_mlp_gated": _own(torch.zeros(total, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "proj_scratch": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
        "proj_scratch2": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, a0=total)

    _single_stream_layer(ctx, fvk, gemm, bufs, ptr_w, dims, 0, 0, 0, attn, mod_single, table.data_ptr())

    cos = _cosine(combined, x_ref)
    print(f"single-stream layer: pointer path vs real reference cosine={cos:.6f}")
    assert cos > 0.999


def test_single_stream_layer_merged_linear1_matches_real_reference():
    """op-fusion audit finding 1 (opportunities.md): `dims["merge_qkv_mlp"]=True`
    path (real fused `linear1.weight`, ONE GEMM for qkv+mlp-gate/up)
    must compute the exact same real math as the split
    `qkv.weight`/`mlp_in.weight` path above -- same real reference, same
    weight VALUES, just concatenated into one wide tensor the way the
    real checkpoint's own unsplit `linear1.weight` already is."""
    cos = _single_stream_layer_merged_vs_reference(merge_linear2=False)
    print(f"single-stream layer (merged linear1): pointer path vs real reference cosine={cos:.6f}")
    assert cos > 0.999


def test_single_stream_layer_merged_linear2_matches_real_reference():
    """Roadmap item 4: `dims["merge_linear2"]=True` (real unsplit
    `linear2.weight`, ONE GEMM over `[attn_out | mlp_act]`) on top of the
    merged `linear1`, against the same tensor-level reference -- the
    weight VALUES are the split test's `attn_out`/`mlp_out` stacked along
    K, the way the real checkpoint stores `linear2.weight`."""
    cos = _single_stream_layer_merged_vs_reference(merge_linear2=True)
    print(f"single-stream layer (merged linear1+linear2): pointer path vs real reference cosine={cos:.6f}")
    assert cos > 0.999


def _single_stream_layer_merged_vs_reference(*, merge_linear2: bool) -> float:
    """Backbone single-stream layer with `merge_qkv_mlp=True` (and
    `merge_linear2` as given) at the small test shape; returns the
    cosine of the pointer path's output against
    `real_single_stream_block_forward_fp16`."""
    NH, HD, mlp_hidden = 2, 128, 192
    hidden = NH * HD
    total = 8
    torch.manual_seed(1)
    scale = 1.0 / (HD ** 0.5)

    gemm = fvk.GemmRunner()

    q, k, v, fused = _fused_qkv(hidden, DEV)
    ref_w = {
        "qkv": torch.cat([q, k, v], dim=1),
        "attn_out": _lin(hidden, hidden, DEV),
        "mlp_in": _lin(mlp_hidden * 2, hidden, DEV),
        "mlp_out": _lin(hidden, mlp_hidden, DEV),
        "query_norm": _norm_scale(HD, DEV),
        "key_norm": _norm_scale(HD, DEV),
    }
    linear1 = _own(torch.cat([ref_w["qkv"], ref_w["mlp_in"]], dim=1).contiguous())
    ptr_w = {
        ("backbone", "single", 0, "linear1.weight"):
            Fp16Linear(gemm, linear1.data_ptr(), 3 * hidden + 2 * mlp_hidden, hidden),
        ("backbone", "single", 0, "query_norm"): ref_w["query_norm"].data_ptr(),
        ("backbone", "single", 0, "key_norm"): ref_w["key_norm"].data_ptr(),
    }
    if merge_linear2:
        linear2 = _own(torch.cat([ref_w["attn_out"], ref_w["mlp_out"]], dim=0).contiguous())
        ptr_w[("backbone", "single", 0, "linear2.weight")] = Fp16Linear(
            gemm, linear2.data_ptr(), hidden, hidden + mlp_hidden)
    else:
        ptr_w[("backbone", "single", 0, "attn_out_proj.weight")] = Fp16Linear(
            gemm, ref_w["attn_out"].data_ptr(), hidden, hidden)
        ptr_w[("backbone", "single", 0, "mlp_down.weight")] = Fp16Linear(
            gemm, ref_w["mlp_out"].data_ptr(), hidden, mlp_hidden)

    mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    timestep = torch.zeros(1, device=DEV)
    _, _, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
    table = build_backbone_rope_table(3, 5, 1, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    ctx = fvk.FvkContext()

    x = _own(torch.randn(total, hidden, dtype=FP16, device=DEV) * 0.1)
    x_ref = real_single_stream_block_forward_fp16(
        gemm, ctx, x.clone(), ref_w, mod_single, table, NH, HD, hidden, mlp_hidden, scale)

    spec = make_imagewam_attention_spec(max_prefix_seq=total, max_total_seq=total + 1,
                                         num_layers=1, num_heads=NH, head_dim=HD)
    K_cache = _own(torch.zeros(1, total, hidden, dtype=FP16, device=DEV))
    V_cache = _own(torch.zeros(1, total, hidden, dtype=FP16, device=DEV))
    Q_O = _own(torch.zeros(total, hidden, dtype=FP16, device=DEV))
    logits = _own(torch.zeros(total * NH, total + (total % 2), dtype=FP16, device=DEV))
    attn = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                        "logits": logits.data_ptr(), "scale": scale},
        mot_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                   "logits": logits.data_ptr(), "scale": scale, "layer_stride": K_cache[0].numel() * 2},
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    combined = _own(x.clone().to(BF16))
    bufs = {
        "backbone_hidden": combined.data_ptr(),
        "modded_scratch": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
        "single_linear1_merged": _own(
            torch.zeros(total, 3 * hidden + 2 * mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "single_linear2_in": _own(torch.zeros(total, hidden + mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "single_mlp_gated": _own(torch.zeros(total, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "proj_scratch": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
        "proj_scratch2": _own(torch.zeros(total, hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, a0=total, merge_qkv_mlp=True,
                merge_linear2=merge_linear2)

    _single_stream_layer(ctx, fvk, gemm, bufs, ptr_w, dims, 0, 0, 0, attn, mod_single, table.data_ptr())

    return _cosine(combined, x_ref)


def _action_common(NH, HD, action_hidden, action_mlp_hidden, backbone_total, device):
    attn_dim = NH * HD
    # ActionDiT's own qkv projects action_hidden -> attn_dim (may
    # differ), so build directly at that shape rather than reusing
    # `_fused_qkv` (which assumes square hidden==attn_dim).
    fused = _own((torch.randn(action_hidden, 3 * attn_dim, dtype=torch.float32, device=device) * 0.02).to(FP16))
    q = fused[:, 0:attn_dim].contiguous()
    k = fused[:, attn_dim:2 * attn_dim].contiguous()
    v = fused[:, 2 * attn_dim:3 * attn_dim].contiguous()
    return q, k, v, fused


def test_action_double_and_single_layers_match_real_reference():
    NH, HD = 2, 128
    attn_dim = NH * HD
    action_hidden, action_mlp_hidden = 96, 128
    backbone_total, num_action = 8, 4
    torch.manual_seed(2)
    scale = 1.0 / (HD ** 0.5)

    cached_k = _own(torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV) * 0.1)
    cached_v = _own(torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV) * 0.1)
    action = _own(torch.randn(num_action, action_hidden, dtype=FP16, device=DEV) * 0.1)

    mod_w = {
        "time_in_w1": torch.randn(action_hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(action_hidden, action_hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * action_hidden, action_hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * action_hidden, action_hidden, dtype=F32, device=DEV) * 0.02,
    }
    timestep = torch.ones(1, device=DEV)
    mod_double, mod_single = compute_action_modulation(timestep, mod_w, action_hidden)
    action_table = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)

    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    total = backbone_total + num_action
    spec = make_imagewam_attention_spec(max_prefix_seq=backbone_total, max_total_seq=total,
                                         num_layers=2, num_heads=NH, head_dim=HD)
    K_cache = _own(torch.zeros(2, total, attn_dim, dtype=FP16, device=DEV))
    V_cache = _own(torch.zeros(2, total, attn_dim, dtype=FP16, device=DEV))
    Q_O = _own(torch.zeros(total, attn_dim, dtype=FP16, device=DEV))
    logits = _own(torch.zeros(num_action * NH, total + (total % 2), dtype=FP16, device=DEV))
    attn = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                        "logits": logits.data_ptr(), "scale": scale},
        mot_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                   "logits": logits.data_ptr(), "scale": scale, "layer_stride": K_cache[0].numel() * 2},
        use_perhead_kv=True, use_real_mot_mask=True,
    )
    # Both "mot" layers (double idx 0, single idx 1) share the SAME
    # frozen backbone cache in this test -- write cached_k/v into both.
    K_cache[0, :backbone_total] = cached_k.reshape(backbone_total, attn_dim)
    V_cache[0, :backbone_total] = cached_v.reshape(backbone_total, attn_dim)
    K_cache[1, :backbone_total] = cached_k.reshape(backbone_total, attn_dim)
    V_cache[1, :backbone_total] = cached_v.reshape(backbone_total, attn_dim)

    dims = dict(action_hidden_dim=action_hidden, action_attn_width=attn_dim, HD=HD, NH=NH,
                action_mlp_hidden=action_mlp_hidden, x0=1, a0=backbone_total, total=total,
                num_action=num_action)

    q, k, v, qkv_fused = _action_common(NH, HD, action_hidden, action_mlp_hidden, backbone_total, DEV)
    ref_w = {
        "qkv": torch.cat([q, k, v], dim=1),
        "proj": _lin(action_hidden, attn_dim, DEV),
        "mlp_in": _lin(action_mlp_hidden * 2, action_hidden, DEV),
        "mlp_out": _lin(action_hidden, action_mlp_hidden, DEV),
        "query_norm": _norm_scale(HD, DEV),
        "key_norm": _norm_scale(HD, DEV),
    }
    ptr_w = {
        ("action_dit", "double", 0, "qkv.weight"): Fp16Linear(gemm, qkv_fused.data_ptr(), 3 * attn_dim, action_hidden),
        ("action_dit", "double", 0, "proj.weight"): Fp16Linear(gemm, ref_w["proj"].data_ptr(), action_hidden, attn_dim),
        ("action_dit", "double", 0, "mlp0.weight"): Fp16Linear(gemm, ref_w["mlp_in"].data_ptr(), action_mlp_hidden * 2, action_hidden),
        ("action_dit", "double", 0, "mlp2.weight"): Fp16Linear(gemm, ref_w["mlp_out"].data_ptr(), action_hidden, action_mlp_hidden),
        ("action_dit", "double", 0, "query_norm"): ref_w["query_norm"].data_ptr(),
        ("action_dit", "double", 0, "key_norm"): ref_w["key_norm"].data_ptr(),
    }
    action_ref = real_action_double_block_forward_fp16(
        gemm, action.clone(), ref_w, mod_double, action_table, cached_k, cached_v,
        NH, HD, action_hidden, action_mlp_hidden, scale)

    action_x = _own(action.clone())
    bufs = {
        "action_hidden": action_x.data_ptr(),
        "action_modded": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_qkv_merged": _own(torch.zeros(num_action, 3 * attn_dim, dtype=FP16, device=DEV)).data_ptr(),
        "action_proj_scratch": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_mlp_merged": _own(torch.zeros(num_action, action_mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
        "action_mlp_gated": _own(torch.zeros(num_action, action_mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    _action_double_layer(ctx, fvk, gemm, bufs, ptr_w, dims, 0, 0, 0, attn, mod_double, action_table.data_ptr())
    cos = _cosine(action_x, action_ref)
    print(f"action double layer: pointer path vs real reference cosine={cos:.6f}")
    assert cos > 0.999

    # Single layer, independent weights, using the SAME cached K/V
    # (site_layer_idx=1, matching K_cache[1] populated above).
    q2, k2, v2, qkv_fused2 = _action_common(NH, HD, action_hidden, action_mlp_hidden, backbone_total, DEV)
    ref_w2 = {
        "qkv": torch.cat([q2, k2, v2], dim=1),
        "attn_out": _lin(action_hidden, attn_dim, DEV),
        "mlp_in": _lin(action_mlp_hidden * 2, action_hidden, DEV),
        "mlp_out": _lin(action_hidden, action_mlp_hidden, DEV),
        "query_norm": _norm_scale(HD, DEV),
        "key_norm": _norm_scale(HD, DEV),
    }
    ptr_w2 = {
        ("action_dit", "single", 0, "qkv.weight"): Fp16Linear(gemm, qkv_fused2.data_ptr(), 3 * attn_dim, action_hidden),
        ("action_dit", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, ref_w2["attn_out"].data_ptr(), action_hidden, attn_dim),
        ("action_dit", "single", 0, "mlp_in.weight"): Fp16Linear(gemm, ref_w2["mlp_in"].data_ptr(), action_mlp_hidden * 2, action_hidden),
        ("action_dit", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, ref_w2["mlp_out"].data_ptr(), action_hidden, action_mlp_hidden),
        ("action_dit", "single", 0, "query_norm"): ref_w2["query_norm"].data_ptr(),
        ("action_dit", "single", 0, "key_norm"): ref_w2["key_norm"].data_ptr(),
    }
    action2_ref = real_action_single_block_forward_fp16(
        gemm, action.clone(), ref_w2, mod_single, action_table, cached_k, cached_v,
        NH, HD, action_hidden, action_mlp_hidden, scale)

    action_x2 = _own(action.clone())
    bufs2 = {
        "action_hidden": action_x2.data_ptr(),
        "action_modded": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_qkv_merged": _own(torch.zeros(num_action, 3 * attn_dim, dtype=FP16, device=DEV)).data_ptr(),
        "action_proj_scratch": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_proj_scratch2": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_mlp_merged": _own(torch.zeros(num_action, action_mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
        "action_mlp_gated": _own(torch.zeros(num_action, action_mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    _action_single_layer(ctx, fvk, gemm, bufs2, ptr_w2, dims, 0, 1, 0, attn, mod_single, action_table.data_ptr())
    cos2 = _cosine(action_x2, action2_ref)
    print(f"action single layer: pointer path vs real reference cosine={cos2:.6f}")
    assert cos2 > 0.999

    # op-fusion audit finding 1 (opportunities.md): same merged-linear1
    # check as test_single_stream_layer_merged_linear1_matches_real_reference,
    # for ActionDiT's own single-stream block -- SAME ref_w2 values
    # (qkv+mlp_in concatenated into one wide tensor), dims["merge_qkv_mlp"]=True,
    # compared against the SAME action2_ref already computed above.
    linear1_action = _own(torch.cat([ref_w2["qkv"], ref_w2["mlp_in"]], dim=1).contiguous())
    ptr_w2_merged = {
        ("action_dit", "single", 0, "linear1.weight"):
            Fp16Linear(gemm, linear1_action.data_ptr(), 3 * attn_dim + 2 * action_mlp_hidden, action_hidden),
        ("action_dit", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, ref_w2["attn_out"].data_ptr(), action_hidden, attn_dim),
        ("action_dit", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, ref_w2["mlp_out"].data_ptr(), action_hidden, action_mlp_hidden),
        ("action_dit", "single", 0, "query_norm"): ref_w2["query_norm"].data_ptr(),
        ("action_dit", "single", 0, "key_norm"): ref_w2["key_norm"].data_ptr(),
    }
    action_x3 = _own(action.clone())
    dims_merged = dict(dims, merge_qkv_mlp=True)
    bufs3 = {
        "action_hidden": action_x3.data_ptr(),
        "action_modded": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_linear1_merged": _own(
            torch.zeros(num_action, 3 * attn_dim + 2 * action_mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_proj_scratch": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_proj_scratch2": _own(torch.zeros(num_action, action_hidden, dtype=FP16, device=DEV)).data_ptr(),
        "action_mlp_gated": _own(torch.zeros(num_action, action_mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
    }
    _action_single_layer(ctx, fvk, gemm, bufs3, ptr_w2_merged, dims_merged, 0, 1, 0, attn, mod_single, action_table.data_ptr())
    cos3 = _cosine(action_x3, action2_ref)
    print(f"action single layer (merged linear1): pointer path vs real reference cosine={cos3:.6f}")
    assert cos3 > 0.999

    # Roadmap item 4: merged linear2 on top of the merged linear1 --
    # SAME ref_w2 values (attn_out/mlp_out stacked along K into one
    # linear2.weight), compared against the SAME action2_ref.
    linear2_action = _own(torch.cat([ref_w2["attn_out"], ref_w2["mlp_out"]], dim=0).contiguous())
    ptr_w2_merged2 = {
        ("action_dit", "single", 0, "linear1.weight"): ptr_w2_merged[("action_dit", "single", 0, "linear1.weight")],
        ("action_dit", "single", 0, "linear2.weight"):
            Fp16Linear(gemm, linear2_action.data_ptr(), action_hidden, attn_dim + action_mlp_hidden),
        ("action_dit", "single", 0, "query_norm"): ref_w2["query_norm"].data_ptr(),
        ("action_dit", "single", 0, "key_norm"): ref_w2["key_norm"].data_ptr(),
    }
    action_x4 = _own(action.clone())
    dims_merged2 = dict(dims, merge_qkv_mlp=True, merge_linear2=True)
    bufs4 = dict(bufs3, action_hidden=action_x4.data_ptr(), action_linear2_in=_own(
        torch.zeros(num_action, attn_dim + action_mlp_hidden, dtype=FP16, device=DEV)).data_ptr())
    _action_single_layer(ctx, fvk, gemm, bufs4, ptr_w2_merged2, dims_merged2, 0, 1, 0, attn, mod_single,
                         action_table.data_ptr())
    cos4 = _cosine(action_x4, action2_ref)
    print(f"action single layer (merged linear1+linear2): pointer path vs real reference cosine={cos4:.6f}")
    assert cos4 > 0.999


# ──────────────────────────────────────────────────────────────────
# Real-shape merged vs split `linear2` (roadmap item 4)
# ──────────────────────────────────────────────────────────────────
_REAL = {k: LIBERO_REAL_DIMS[k] for k in (
    "NH", "HD", "hidden", "mlp_hidden", "x0", "a0",
    "action_hidden_dim", "action_attn_width", "action_mlp_hidden", "num_action")}


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """cosine / max-abs / rel_l2 of `a` against `b`, plus whether they
    are bitwise equal."""
    a_, b_ = a.float().flatten(), b.float().flatten()
    return dict(cos=_cosine(a_, b_), max_abs=(a_ - b_).abs().max().item(),
                rel_l2=((a_ - b_).norm() / (b_.norm() + 1e-12)).item(), bit_exact=torch.equal(a, b))


def _fmt(st: dict) -> str:
    return (f"cos={st['cos']:.7f} max_abs={st['max_abs']:.3e} rel_l2={st['rel_l2']:.3e} "
            f"bit_exact={st['bit_exact']}")


def _real_attn(seq_rows: int, total: int, num_layers: int, ctx, *, prefill_kv: bool):
    """ImageWAMAttnBackend at the real head geometry. `prefill_kv`
    fills rows `[0, a0)` of every layer's K/V with random values (the
    frozen backbone cache the "mot" site attends into)."""
    NH, HD, hidden, a0 = _REAL["NH"], _REAL["HD"], _REAL["hidden"], _REAL["a0"]
    scale = 1.0 / (HD ** 0.5)
    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total,
                                         num_layers=num_layers, num_heads=NH, head_dim=HD)
    K_cache = _own(torch.zeros(num_layers, total, hidden, dtype=FP16, device=DEV))
    V_cache = _own(torch.zeros(num_layers, total, hidden, dtype=FP16, device=DEV))
    if prefill_kv:
        K_cache[:, :a0] = torch.randn(num_layers, a0, hidden, dtype=FP16, device=DEV)
        V_cache[:, :a0] = torch.randn(num_layers, a0, hidden, dtype=FP16, device=DEV)
    Q_O = _own(torch.zeros(total, hidden, dtype=FP16, device=DEV))
    logits = _own(torch.zeros(seq_rows * NH, total + (total % 2), dtype=FP16, device=DEV))
    slots = {"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
             "logits": logits.data_ptr(), "scale": scale}
    return ImageWAMAttnBackend(spec, ctx, backbone_slots=dict(slots),
                               mot_slots=dict(slots, layer_stride=K_cache[0].numel() * 2),
                               use_perhead_kv=True, use_real_mot_mask=True)


def test_single_stream_linear2_merged_vs_split_real_shapes():
    """Roadmap item 4 at the REAL shapes: the same random weights run
    through `_single_stream_layer` (backbone, a0=905, hidden 3072, mlp
    9216) and `_action_single_layer` (M=64, 1024/3072/4096) with the
    split `attn_out_proj`+`mlp_down` path and the merged `linear2` path
    (both on the merged `linear1`). Reports the projection feeding the
    gated residual (`proj_scratch`, the only quantity the merge changes)
    and the layer output."""
    R = _REAL
    NH, HD, hidden, mlp_hidden, a0 = R["NH"], R["HD"], R["hidden"], R["mlp_hidden"], R["a0"]
    torch.manual_seed(5)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    # --- backbone single-stream layer ---
    l1_w = _own((torch.randn(hidden, 3 * hidden + 2 * mlp_hidden, device=DEV) * 0.02).to(FP16))
    attn_out = _own((torch.randn(hidden, hidden, device=DEV) * 0.02).to(FP16))
    mlp_down = _own((torch.randn(mlp_hidden, hidden, device=DEV) * 0.02).to(FP16))
    linear2 = _own(torch.cat([attn_out, mlp_down], dim=0).contiguous())
    common = {
        ("backbone", "single", 0, "linear1.weight"): Fp16Linear(gemm, l1_w.data_ptr(), 3 * hidden + 2 * mlp_hidden, hidden),
        ("backbone", "single", 0, "query_norm"): _norm_scale(HD, DEV).data_ptr(),
        ("backbone", "single", 0, "key_norm"): _norm_scale(HD, DEV).data_ptr(),
    }
    w_split = {**common,
        ("backbone", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, attn_out.data_ptr(), hidden, hidden),
        ("backbone", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, mlp_down.data_ptr(), hidden, mlp_hidden),
    }
    w_merged = {**common,
        ("backbone", "single", 0, "linear2.weight"): Fp16Linear(gemm, linear2.data_ptr(), hidden, hidden + mlp_hidden),
    }
    mod_w = {
        "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
    }
    _, _, mod_single = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, hidden)
    table = build_backbone_rope_table(R["x0"], 14, 28, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    attn = _real_attn(a0, a0, 1, ctx, prefill_kv=False)
    x_in = _own((torch.randn(a0, hidden, device=DEV) * 4.0).to(BF16))

    outs = {}
    l2_in = _own(torch.zeros(a0, hidden + mlp_hidden, dtype=FP16, device=DEV))
    for name, w, merged in (("split", w_split, False), ("merged", w_merged, True)):
        combined = _own(x_in.clone())
        proj = _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV))
        bufs = {
            "backbone_hidden": combined.data_ptr(),
            "modded_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_linear1_merged": _own(torch.zeros(a0, 3 * hidden + 2 * mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_linear2_in": l2_in.data_ptr(),
            "single_mlp_gated": _own(torch.zeros(a0, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch": proj.data_ptr(),
            "proj_scratch2": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, a0=a0, merge_qkv_mlp=True,
                    merge_linear2=merged)
        _single_stream_layer(ctx, fvk, gemm, bufs, w, dims, 0, 0, 0, attn, mod_single, table.data_ptr())
        torch.cuda.synchronize()
        outs[name] = (proj.clone(), combined.clone())
    proj_st = _diff_stats(outs["merged"][0], outs["split"][0])
    out_st = _diff_stats(outs["merged"][1], outs["split"][1])
    # FP32 reference for the projection over the SAME [attn_out | mlp_act]
    # operands (the merged run's own linear2 input buffer).
    ref32 = l2_in.float() @ linear2.float()
    print(f"backbone single (a0={a0}) merged vs split linear2: proj {_fmt(proj_st)}")
    print(f"backbone single (a0={a0}) merged vs split linear2: layer output {_fmt(out_st)}")
    print(f"backbone single proj rel_l2 vs fp32 reference: merged={_diff_stats(outs['merged'][0], ref32)['rel_l2']:.3e} "
          f"split={_diff_stats(outs['split'][0], ref32)['rel_l2']:.3e}")
    assert proj_st["cos"] > 0.9999 and out_st["cos"] > 0.9999

    # --- ActionDiT single-stream layer ---
    ahd, aaw, amh, num_action = R["action_hidden_dim"], R["action_attn_width"], R["action_mlp_hidden"], R["num_action"]
    total = a0 + num_action
    al1_w = _own((torch.randn(ahd, 3 * aaw + 2 * amh, device=DEV) * 0.02).to(FP16))
    a_attn_out = _own((torch.randn(aaw, ahd, device=DEV) * 0.02).to(FP16))
    a_mlp_down = _own((torch.randn(amh, ahd, device=DEV) * 0.02).to(FP16))
    a_linear2 = _own(torch.cat([a_attn_out, a_mlp_down], dim=0).contiguous())
    a_common = {
        ("action_dit", "single", 0, "linear1.weight"): Fp16Linear(gemm, al1_w.data_ptr(), 3 * aaw + 2 * amh, ahd),
        ("action_dit", "single", 0, "query_norm"): _norm_scale(HD, DEV).data_ptr(),
        ("action_dit", "single", 0, "key_norm"): _norm_scale(HD, DEV).data_ptr(),
    }
    a_split = {**a_common,
        ("action_dit", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, a_attn_out.data_ptr(), ahd, aaw),
        ("action_dit", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, a_mlp_down.data_ptr(), ahd, amh),
    }
    a_merged = {**a_common,
        ("action_dit", "single", 0, "linear2.weight"): Fp16Linear(gemm, a_linear2.data_ptr(), ahd, aaw + amh),
    }
    amod_w = {
        "time_in_w1": torch.randn(ahd, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
    }
    _, amod_single = compute_action_modulation(torch.full((1,), 0.5, device=DEV), amod_w, ahd)
    atable = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    aattn = _real_attn(num_action, total, 1, ctx, prefill_kv=True)
    a_in = _own(torch.randn(num_action, ahd, dtype=FP16, device=DEV))

    aouts = {}
    a_l2_in = _own(torch.zeros(num_action, aaw + amh, dtype=FP16, device=DEV))
    for name, w, merged in (("split", a_split, False), ("merged", a_merged, True)):
        action_x = _own(a_in.clone())
        proj = _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV))
        bufs = {
            "action_hidden": action_x.data_ptr(),
            "action_modded": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
            "action_linear1_merged": _own(torch.zeros(num_action, 3 * aaw + 2 * amh, dtype=FP16, device=DEV)).data_ptr(),
            "action_linear2_in": a_l2_in.data_ptr(),
            "action_mlp_gated": _own(torch.zeros(num_action, amh, dtype=FP16, device=DEV)).data_ptr(),
            "action_proj_scratch": proj.data_ptr(),
            "action_proj_scratch2": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(action_hidden_dim=ahd, action_attn_width=aaw, HD=HD, NH=NH, action_mlp_hidden=amh,
                    x0=R["x0"], a0=a0, total=total, num_action=num_action, merge_qkv_mlp=True,
                    merge_linear2=merged)
        _action_single_layer(ctx, fvk, gemm, bufs, w, dims, 0, 0, 0, aattn, amod_single, atable.data_ptr())
        torch.cuda.synchronize()
        aouts[name] = (proj.clone(), action_x.clone())
    aproj_st = _diff_stats(aouts["merged"][0], aouts["split"][0])
    aout_st = _diff_stats(aouts["merged"][1], aouts["split"][1])
    print(f"action single (M={num_action}) merged vs split linear2: proj {_fmt(aproj_st)}")
    print(f"action single (M={num_action}) merged vs split linear2: layer output {_fmt(aout_st)}")
    aref32 = a_l2_in.float() @ a_linear2.float()
    print(f"action single proj rel_l2 vs fp32 reference: merged={_diff_stats(aouts['merged'][0], aref32)['rel_l2']:.3e} "
          f"split={_diff_stats(aouts['split'][0], aref32)['rel_l2']:.3e}")
    assert aproj_st["cos"] > 0.9999 and aout_st["cos"] > 0.9999


def test_single_stream_and_action_single_fuse_qkv_norm_rope_bit_exact_at_real_shapes():
    """OPT-032 candidate 1 (`dims["fuse_qkv_norm_rope"]`), wired into
    `_single_stream_layer` and `_action_single_layer`: `fvk.qkv_split_norm_rope_fp16`
    (`csrc/kernels/fused_qkv_norm_rope/`) must give the EXACT same
    layer output as the unfused 3-copy + 2-rms_norm + 2-rope sequence it
    replaces, at real production shapes (backbone a0=905,
    hidden=3072/mlp=9216; ActionDiT M=64, 1024/3072/4096), both on the
    `merge_qkv_mlp=True` path (the real deployment default) and the
    split `qkv.weight` path. `torch.equal`, not cosine: the kernel was
    already proven bit-exact in isolation (this file's other tests use
    cosine because DIFFERENT weight layouts are being compared here; this
    test holds every input identical and only flips `fuse_qkv_norm_rope`,
    so anything short of bit-exact is a wiring bug -- a wrong row/column
    offset into the merged buffer, not a numerics difference)."""
    R = _REAL
    NH, HD, hidden, mlp_hidden, a0 = R["NH"], R["HD"], R["hidden"], R["mlp_hidden"], R["a0"]
    torch.manual_seed(11)
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    def build_single_stream(*, merge_qkv_mlp: bool):
        """Weights/modulation/input/attn built ONCE, reused for both the
        `fused_qkv=False` and `=True` replay of the SAME layer call --
        `run_single_stream` below only swaps `dims["fuse_qkv_norm_rope"]`
        and re-runs against fresh (but identically-seeded) buffers.

        Rebuilding fresh `Fp16Linear` weight tensors for each of the two
        compared calls (this function's earlier design, and the seed-reset
        bug fixed above) hits a SEPARATE, pre-existing issue this file's own
        debugging found: calling one of these layer functions twice against
        one shared `GemmRunner`, with different weight tensor pointers at
        the SAME (M,N,K) shape each time, corrupts something the LATER
        cuBLASLt call then trips over (`RuntimeError: cuBLAS error ...
        code=13`), reproduced locally even with `fuse_qkv_norm_rope=False`
        on BOTH calls -- i.e. unrelated to this kernel. Building the
        weights once and replaying against them (matching
        `test_single_stream_linear2_merged_vs_split_real_shapes`'s own
        established pattern) avoids it entirely; see issues.md.
        """
        torch.manual_seed(11)
        q_norm, k_norm = _norm_scale(HD, DEV), _norm_scale(HD, DEV)
        w = {
            ("backbone", "single", 0, "query_norm"): q_norm.data_ptr(),
            ("backbone", "single", 0, "key_norm"): k_norm.data_ptr(),
        }
        if merge_qkv_mlp:
            l1_w = _own((torch.randn(hidden, 3 * hidden + 2 * mlp_hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "single", 0, "linear1.weight")] = Fp16Linear(
                gemm, l1_w.data_ptr(), 3 * hidden + 2 * mlp_hidden, hidden)
        else:
            qkv_w = _own((torch.randn(hidden, 3 * hidden, device=DEV) * 0.02).to(FP16))
            mlp_in_w = _own((torch.randn(hidden, 2 * mlp_hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "single", 0, "qkv.weight")] = Fp16Linear(gemm, qkv_w.data_ptr(), 3 * hidden, hidden)
            w[("backbone", "single", 0, "mlp_in.weight")] = Fp16Linear(gemm, mlp_in_w.data_ptr(), 2 * mlp_hidden, hidden)
        attn_out = _own((torch.randn(hidden, hidden, device=DEV) * 0.02).to(FP16))
        mlp_down = _own((torch.randn(mlp_hidden, hidden, device=DEV) * 0.02).to(FP16))
        w[("backbone", "single", 0, "attn_out_proj.weight")] = Fp16Linear(gemm, attn_out.data_ptr(), hidden, hidden)
        w[("backbone", "single", 0, "mlp_down.weight")] = Fp16Linear(gemm, mlp_down.data_ptr(), hidden, mlp_hidden)

        mod_w = {
            "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
            "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        }
        _, _, mod_single = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, hidden)
        table = build_backbone_rope_table(R["x0"], 14, 28, axes_dim=AXES_DIM, theta=THETA, device=DEV)
        attn = _real_attn(a0, a0, 1, ctx, prefill_kv=False)
        x_in = _own((torch.randn(a0, hidden, device=DEV) * 4.0).to(BF16))
        return w, mod_single, table, attn, x_in

    def run_single_stream(built, *, merge_qkv_mlp: bool, fused_qkv: bool) -> torch.Tensor:
        w, mod_single, table, attn, x_in = built
        combined = _own(x_in.clone())
        bufs = {
            "backbone_hidden": combined.data_ptr(),
            "modded_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_qkv_merged": _own(torch.zeros(a0, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_mlp_merged": _own(torch.zeros(a0, 2 * mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_linear1_merged": _own(torch.zeros(a0, 3 * hidden + 2 * mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_mlp_gated": _own(torch.zeros(a0, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch2": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, a0=a0,
                    merge_qkv_mlp=merge_qkv_mlp, merge_linear2=False, fuse_qkv_norm_rope=fused_qkv)
        _single_stream_layer(ctx, fvk, gemm, bufs, w, dims, 0, 0, 0, attn, mod_single, table.data_ptr())
        torch.cuda.synchronize()
        return combined.clone()

    for merge_qkv_mlp in (True, False):
        built = build_single_stream(merge_qkv_mlp=merge_qkv_mlp)
        unfused = run_single_stream(built, merge_qkv_mlp=merge_qkv_mlp, fused_qkv=False)
        fused = run_single_stream(built, merge_qkv_mlp=merge_qkv_mlp, fused_qkv=True)
        st = _diff_stats(fused, unfused)
        print(f"backbone single (merge_qkv_mlp={merge_qkv_mlp}) fused vs unfused qkv_norm_rope: {_fmt(st)}")
        assert torch.equal(fused, unfused), f"merge_qkv_mlp={merge_qkv_mlp}: not bit-exact, {_fmt(st)}"

    ahd, aaw, amh, num_action = R["action_hidden_dim"], R["action_attn_width"], R["action_mlp_hidden"], R["num_action"]
    total = a0 + num_action

    # Weights/modulation/attn/input built ONCE and replayed for both
    # fused_qkv values -- see build_single_stream's own docstring above for
    # why (a pre-existing GemmRunner/repeated-fresh-weight issue, not a
    # fuse_qkv_norm_rope bug: this exact ActionDiT shape is where it was
    # actually found, issues.md).
    torch.manual_seed(13)
    q_norm, k_norm = _norm_scale(HD, DEV), _norm_scale(HD, DEV)
    al1_w = _own((torch.randn(ahd, 3 * aaw + 2 * amh, device=DEV) * 0.02).to(FP16))
    a_attn_out = _own((torch.randn(aaw, ahd, device=DEV) * 0.02).to(FP16))
    a_mlp_down = _own((torch.randn(amh, ahd, device=DEV) * 0.02).to(FP16))
    aw = {
        ("action_dit", "single", 0, "linear1.weight"): Fp16Linear(gemm, al1_w.data_ptr(), 3 * aaw + 2 * amh, ahd),
        ("action_dit", "single", 0, "query_norm"): q_norm.data_ptr(),
        ("action_dit", "single", 0, "key_norm"): k_norm.data_ptr(),
        ("action_dit", "single", 0, "attn_out_proj.weight"): Fp16Linear(gemm, a_attn_out.data_ptr(), aaw, ahd),
        ("action_dit", "single", 0, "mlp_down.weight"): Fp16Linear(gemm, a_mlp_down.data_ptr(), amh, ahd),
    }
    amod_w = {
        "time_in_w1": torch.randn(ahd, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
    }
    _, amod_single = compute_action_modulation(torch.full((1,), 0.5, device=DEV), amod_w, ahd)
    atable = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    aattn = _real_attn(num_action, total, 1, ctx, prefill_kv=True)
    a_in = _own(torch.randn(num_action, ahd, dtype=FP16, device=DEV))

    def run_action_single(*, fused_qkv: bool) -> torch.Tensor:
        action_x = _own(a_in.clone())
        bufs = {
            "action_hidden": action_x.data_ptr(),
            "action_modded": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
            "action_linear1_merged": _own(torch.zeros(num_action, 3 * aaw + 2 * amh, dtype=FP16, device=DEV)).data_ptr(),
            "action_linear2_in": _own(torch.zeros(num_action, aaw + amh, dtype=FP16, device=DEV)).data_ptr(),
            "action_mlp_gated": _own(torch.zeros(num_action, amh, dtype=FP16, device=DEV)).data_ptr(),
            "action_proj_scratch": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
            "action_proj_scratch2": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(action_hidden_dim=ahd, action_attn_width=aaw, HD=HD, NH=NH, action_mlp_hidden=amh,
                    x0=R["x0"], a0=a0, total=total, num_action=num_action, merge_qkv_mlp=True,
                    merge_linear2=False, fuse_qkv_norm_rope=fused_qkv)
        _action_single_layer(ctx, fvk, gemm, bufs, aw, dims, 0, 0, 0, aattn, amod_single, atable.data_ptr())
        torch.cuda.synchronize()
        return action_x.clone()

    a_unfused = run_action_single(fused_qkv=False)
    a_fused = run_action_single(fused_qkv=True)
    a_st = _diff_stats(a_fused, a_unfused)
    print(f"action single fused vs unfused qkv_norm_rope: {_fmt(a_st)}")
    assert torch.equal(a_fused, a_unfused), f"not bit-exact, {_fmt(a_st)}"


def test_double_stream_and_action_double_fuse_qkv_norm_rope_bit_exact_at_real_shapes():
    """OPT-032 candidate 1, Phase 2 (`dims["fuse_qkv_norm_rope"]`), wired
    into `_double_stream_layer` and `_action_double_layer`: same
    bit-exact contract as Phase 1's single-stream test above, at real
    production shapes (backbone x0=513/img_len=392/a0=905,
    hidden=3072/mlp=9216; ActionDiT num_action=64,
    action_hidden_dim/action_attn_width/action_mlp_hidden from `_REAL`).
    `_double_stream_layer` runs RMSNorm separately per stream (txt then
    img) so it calls the fused kernel TWICE, once per stream, each with
    its own row-offset rope_table pointer, instead of ONE joint
    whole-sequence RoPE call -- proven row-independent-equal to that
    joint call by tests/test_fused_qkv_norm_rope_kernel.py::
    test_double_stream_split_call_matches_joint_rope; this test checks
    the actual PRODUCTION wiring at real shapes, not a standalone
    repro. `_action_double_layer` is a single-call site, same pattern
    as `_action_single_layer`. Builds weights/attn/input ONCE and
    replays both `fuse_qkv_norm_rope` values against them (issues.md
    ISSUE-090, same reason as Phase 1's test above)."""
    R = _REAL
    NH, HD, hidden, mlp_hidden = R["NH"], R["HD"], R["hidden"], R["mlp_hidden"]
    x0, a0 = R["x0"], R["a0"]
    img_len = a0 - x0
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    def build_double_stream():
        torch.manual_seed(17)
        w = {}
        for side in ("txt", "img"):
            qkv_w = _own((torch.randn(hidden, 3 * hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "double", 0, f"{side}_qkv.weight")] = Fp16Linear(gemm, qkv_w.data_ptr(), 3 * hidden, hidden)
            proj_w = _own((torch.randn(hidden, hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "double", 0, f"{side}_proj.weight")] = Fp16Linear(gemm, proj_w.data_ptr(), hidden, hidden)
            mlp0_w = _own((torch.randn(hidden, 2 * mlp_hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "double", 0, f"{side}_mlp0.weight")] = Fp16Linear(gemm, mlp0_w.data_ptr(), 2 * mlp_hidden, hidden)
            mlp2_w = _own((torch.randn(mlp_hidden, hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "double", 0, f"{side}_mlp2.weight")] = Fp16Linear(gemm, mlp2_w.data_ptr(), hidden, mlp_hidden)
            w[("backbone", "double", 0, f"{side}_query_norm")] = _norm_scale(HD, DEV).data_ptr()
            w[("backbone", "double", 0, f"{side}_key_norm")] = _norm_scale(HD, DEV).data_ptr()

        mod_w = {
            "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
            "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        }
        mod_txt, mod_img, _ = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, hidden)
        table = build_backbone_rope_table(x0, 14, 28, axes_dim=AXES_DIM, theta=THETA, device=DEV)
        attn = _real_attn(a0, a0, 1, ctx, prefill_kv=False)
        x_in = _own((torch.randn(a0, hidden, device=DEV) * 4.0).to(BF16))
        return w, mod_txt, mod_img, table, attn, x_in

    def run_double_stream(built, *, fused_qkv: bool) -> torch.Tensor:
        w, mod_txt, mod_img, table, attn, x_in = built
        combined = _own(x_in.clone())
        bufs = {
            "backbone_hidden": combined.data_ptr(),
            "modded_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "txt_qkv_merged": _own(torch.zeros(x0, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
            "img_qkv_merged": _own(torch.zeros(img_len, 3 * hidden, dtype=FP16, device=DEV)).data_ptr(),
            "txt_mlp_merged": _own(torch.zeros(x0, mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
            "txt_mlp_gated": _own(torch.zeros(x0, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "img_mlp_merged": _own(torch.zeros(img_len, mlp_hidden * 2, dtype=FP16, device=DEV)).data_ptr(),
            "img_mlp_gated": _own(torch.zeros(img_len, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, x0=x0, a0=a0,
                    fuse_qkv_norm_rope=fused_qkv)
        _double_stream_layer(ctx, fvk, gemm, bufs, w, dims, 0, 0, attn, mod_txt, mod_img, table.data_ptr())
        torch.cuda.synchronize()
        return combined.clone()

    built = build_double_stream()
    unfused = run_double_stream(built, fused_qkv=False)
    fused = run_double_stream(built, fused_qkv=True)
    st = _diff_stats(fused, unfused)
    print(f"backbone double-stream fused vs unfused qkv_norm_rope: {_fmt(st)}")
    assert torch.equal(fused, unfused), f"not bit-exact, {_fmt(st)}"

    # --- ActionDiT double block (img-only, single fused-kernel call per layer) ---
    ahd, aaw, amh, num_action = R["action_hidden_dim"], R["action_attn_width"], R["action_mlp_hidden"], R["num_action"]
    total = a0 + num_action
    torch.manual_seed(19)
    q_norm, k_norm = _norm_scale(HD, DEV), _norm_scale(HD, DEV)
    qkv_w = _own((torch.randn(ahd, 3 * aaw, device=DEV) * 0.02).to(FP16))
    proj_w = _own((torch.randn(aaw, ahd, device=DEV) * 0.02).to(FP16))
    mlp0_w = _own((torch.randn(ahd, 2 * amh, device=DEV) * 0.02).to(FP16))
    mlp2_w = _own((torch.randn(amh, ahd, device=DEV) * 0.02).to(FP16))
    aw = {
        ("action_dit", "double", 0, "qkv.weight"): Fp16Linear(gemm, qkv_w.data_ptr(), 3 * aaw, ahd),
        ("action_dit", "double", 0, "proj.weight"): Fp16Linear(gemm, proj_w.data_ptr(), ahd, aaw),
        ("action_dit", "double", 0, "mlp0.weight"): Fp16Linear(gemm, mlp0_w.data_ptr(), 2 * amh, ahd),
        ("action_dit", "double", 0, "mlp2.weight"): Fp16Linear(gemm, mlp2_w.data_ptr(), ahd, amh),
        ("action_dit", "double", 0, "query_norm"): q_norm.data_ptr(),
        ("action_dit", "double", 0, "key_norm"): k_norm.data_ptr(),
    }
    amod_w = {
        "time_in_w1": torch.randn(ahd, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * ahd, ahd, dtype=F32, device=DEV) * 0.02,
    }
    amod_double, _ = compute_action_modulation(torch.full((1,), 0.5, device=DEV), amod_w, ahd)
    atable = build_action_rope_table(num_action, axes_dim=AXES_DIM, theta=THETA, device=DEV)
    aattn = _real_attn(num_action, total, 1, ctx, prefill_kv=True)
    a_in = _own(torch.randn(num_action, ahd, dtype=FP16, device=DEV))

    def run_action_double(*, fused_qkv: bool) -> torch.Tensor:
        action_x = _own(a_in.clone())
        bufs = {
            "action_hidden": action_x.data_ptr(),
            "action_modded": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
            "action_qkv_merged": _own(torch.zeros(num_action, 3 * aaw, dtype=FP16, device=DEV)).data_ptr(),
            "action_proj_scratch": _own(torch.zeros(num_action, ahd, dtype=FP16, device=DEV)).data_ptr(),
            "action_mlp_merged": _own(torch.zeros(num_action, amh * 2, dtype=FP16, device=DEV)).data_ptr(),
            "action_mlp_gated": _own(torch.zeros(num_action, amh, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(action_hidden_dim=ahd, action_attn_width=aaw, HD=HD, NH=NH, action_mlp_hidden=amh,
                    x0=x0, a0=a0, total=total, num_action=num_action, fuse_qkv_norm_rope=fused_qkv)
        _action_double_layer(ctx, fvk, gemm, bufs, aw, dims, 0, 0, 0, aattn, amod_double, atable.data_ptr())
        torch.cuda.synchronize()
        return action_x.clone()

    a_unfused = run_action_double(fused_qkv=False)
    a_fused = run_action_double(fused_qkv=True)
    a_st = _diff_stats(a_fused, a_unfused)
    print(f"action double fused vs unfused qkv_norm_rope: {_fmt(a_st)}")
    assert torch.equal(a_fused, a_unfused), f"not bit-exact, {_fmt(a_st)}"


def test_single_stream_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes():
    """OPT-032 candidate 3, Phase 4 Round 1 (`dims["fuse_res_norm_fp4"]`):
    a two-layer single-stream chain (layer 0 -> layer 1, `fuse_res_norm`
    already on) where layer 1's `linear1.weight` is a real `Nvfp4Linear`.
    `fuse_res_norm_fp4=False` is today's already-wired path: layer 0's
    `_fused_gate_res` writes a plain FP16 AdaLN into `modded`, layer 1's
    `linear1(modded, ...)` (`Nvfp4Linear.__call__`) quantizes it then
    runs the GEMM. `fuse_res_norm_fp4=True` skips the FP16 intermediate:
    layer 0 writes NVFP4+SFA straight into layer 1's own `linear1`
    activation scratch (`csrc/kernels/fused_norm_fp4/`, already
    Thor-confirmed bit-exact against the unfused pair, THOR_CHECKLIST.md
    X6), and layer 1 reads it via `linear1.gemm_prequantized(...)`
    instead of `__call__`. `torch.equal`, not cosine: holds every input
    identical and only flips `fuse_res_norm_fp4`, so anything short of
    bit-exact is a wiring bug, not a numerics difference.

    Needs a Blackwell/Thor NVFP4 build (`flash_rt.flash_rt_fp4`,
    `-DGPU_ARCH=110`/`ENABLE_SM100_CUTLASS`) -- skipped on any machine
    without one, including this dev machine (Ada, sm_89): `Nvfp4Linear`
    itself cannot be constructed there. See `_fused_gate_res`'s and
    `Nvfp4Linear.gemm_prequantized`'s own docstrings for the contract
    this checks; those and the wiring in `_single_stream_layer` were
    reviewed against the already-Thor-confirmed fused kernel and the
    existing bit-exact call-order dispatch test
    (`tests/test_imagewam_fuse_res_norm_fp4_dispatch.py`, CPU-only), but
    this specific two-layer chain has not been run anywhere before this
    Thor round."""
    pytest.importorskip("flash_rt.flash_rt_fp4")
    R = _REAL
    NH, HD, hidden, mlp_hidden, a0 = R["NH"], R["HD"], R["hidden"], R["mlp_hidden"], R["a0"]
    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()

    def build_chain():
        torch.manual_seed(31)
        w = {}
        for i, linear1_lin in enumerate(("fp16", "nvfp4")):
            l1_w = _own((torch.randn(hidden, 3 * hidden + 2 * mlp_hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "single", i, "linear1.weight")] = (
                Fp16Linear(gemm, l1_w.data_ptr(), 3 * hidden + 2 * mlp_hidden, hidden) if linear1_lin == "fp16"
                else Nvfp4Linear(l1_w.data_ptr(), 3 * hidden + 2 * mlp_hidden, hidden))
            attn_out = _own((torch.randn(hidden, hidden, device=DEV) * 0.02).to(FP16))
            mlp_down = _own((torch.randn(mlp_hidden, hidden, device=DEV) * 0.02).to(FP16))
            w[("backbone", "single", i, "attn_out_proj.weight")] = Fp16Linear(gemm, attn_out.data_ptr(), hidden, hidden)
            w[("backbone", "single", i, "mlp_down.weight")] = Fp16Linear(gemm, mlp_down.data_ptr(), hidden, mlp_hidden)
            w[("backbone", "single", i, "query_norm")] = _norm_scale(HD, DEV).data_ptr()
            w[("backbone", "single", i, "key_norm")] = _norm_scale(HD, DEV).data_ptr()

        mod_w = {
            "time_in_w1": torch.randn(hidden, 256, dtype=F32, device=DEV) * 0.02,
            "time_in_w2": torch.randn(hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_txt": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_double_img": torch.randn(6 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
            "mod_single": torch.randn(3 * hidden, hidden, dtype=F32, device=DEV) * 0.02,
        }
        _, _, mod_single = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, hidden)
        table = build_backbone_rope_table(R["x0"], 14, 28, axes_dim=AXES_DIM, theta=THETA, device=DEV)
        attn = _real_attn(a0, a0, 2, ctx, prefill_kv=False)  # two site layers now, 0 and 1
        x_in = _own((torch.randn(a0, hidden, device=DEV) * 4.0).to(BF16))
        return w, mod_single, table, attn, x_in

    def run_chain(built, *, fuse_res_norm_fp4: bool) -> torch.Tensor:
        w, mod_single, table, attn, x_in = built
        combined = _own(x_in.clone())
        bufs = {
            "backbone_hidden": combined.data_ptr(),
            "modded_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_linear1_merged": _own(torch.zeros(a0, 3 * hidden + 2 * mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "single_mlp_gated": _own(torch.zeros(a0, mlp_hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
            "proj_scratch2": _own(torch.zeros(a0, hidden, dtype=FP16, device=DEV)).data_ptr(),
        }
        dims = dict(hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden, a0=a0,
                    merge_qkv_mlp=True, merge_linear2=False, fuse_res_norm=True,
                    fuse_res_norm_fp4=fuse_res_norm_fp4)
        shift, scale, _gate = mod_single
        next_norm = _awq_target(w[("backbone", "single", 1, "linear1.weight")],
                                AdaLNTarget(shift, scale, bufs["modded_scratch"]))
        _single_stream_layer(ctx, fvk, gemm, bufs, w, dims, 0, 0, 0, attn, mod_single, table.data_ptr(),
                              input_normed=False, next_norm=next_norm)
        _single_stream_layer(ctx, fvk, gemm, bufs, w, dims, 1, 1, 0, attn, mod_single, table.data_ptr(),
                              input_normed=True, next_norm=None)
        torch.cuda.synchronize()
        return combined.clone()

    built = build_chain()
    unfused = run_chain(built, fuse_res_norm_fp4=False)
    fused = run_chain(built, fuse_res_norm_fp4=True)
    st = _diff_stats(fused, unfused)
    print(f"single-stream chain fuse_res_norm_fp4 direct vs quantize-after: {_fmt(st)}")
    assert torch.equal(fused, unfused), f"not bit-exact, {_fmt(st)}"


if __name__ == "__main__":
    test_double_stream_layer_matches_real_reference()
    test_single_stream_layer_matches_real_reference()
    test_single_stream_layer_merged_linear1_matches_real_reference()
    test_single_stream_layer_merged_linear2_matches_real_reference()
    test_action_double_and_single_layers_match_real_reference()
    test_single_stream_linear2_merged_vs_split_real_shapes()
    test_single_stream_and_action_single_fuse_qkv_norm_rope_bit_exact_at_real_shapes()
    test_double_stream_and_action_double_fuse_qkv_norm_rope_bit_exact_at_real_shapes()
    test_single_stream_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes()
    print("PASS")

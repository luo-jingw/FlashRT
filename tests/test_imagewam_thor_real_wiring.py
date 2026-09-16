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
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_real import compute_action_modulation, compute_shared_modulation
from flash_rt.models.imagewam.pipeline_thor import _action_double_layer, _action_single_layer, _double_stream_layer, _single_stream_layer
from flash_rt.models.imagewam.quant_linear import Fp16Linear
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


if __name__ == "__main__":
    test_double_stream_layer_matches_real_reference()
    test_single_stream_layer_matches_real_reference()
    test_action_double_and_single_layers_match_real_reference()
    print("PASS")

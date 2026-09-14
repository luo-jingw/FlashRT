"""ImageWAM (FLUX.2-4B variant) Thor compute path — backbone prefill +
ActionDiT denoise loop, REAL math (opportunities.md OPT-002).

**Rewritten 2026-09-14** to replace the earlier structural-dry-run
approximation (unweighted RMS norm, single-shared-broadcast K/V, no
RoPE/QK-Norm/AdaLN, plain GELU MLP at the wrong width, and a wrong
attention-mask exclusion) with the real math this project has since
verified end to end — including against the real trained checkpoint on
Thor (`benchmarks/imagewam_real_checkpoint_validation.py`, backbone
cosine=0.999927, ActionDiT cosine=0.999963). This is now the confirmed
target for real Thor deployment (`PROJECT.md`'s "Confirmed end goal"),
not a permanent parallel/throwaway path — see `opportunities.md` OPT-002
for the full history of what was found and fixed.

Weights are still random-initialized here (see
`flash_rt/frontends/torch/imagewam_thor.py`) — real checkpoint loading
is a separate, still-open piece of work (needs the real `imagewam`/
`flux2` packages, only available on Thor; see
`benchmarks/imagewam_real_checkpoint_validation.py` for the weight
EXTRACTION logic already verified against the real checkpoint, which a
future loader would reuse). Nothing about the math below depends on
where the weights come from.

# Real per-layer structure
=======================================================================

Ported from `flash_rt/models/imagewam/real_double_stream_block.py` /
`real_single_stream_block.py` / `real_action_expert.py` (which remain
the verified tensor-level reference this pointer-based rewrite is
checked against — see `tests/test_imagewam_thor_real_wiring.py`), but
restructured as pointer-based, buffer-reusing code with attention
dispatched through `ImageWAMAttnBackend` (constructed with
`use_perhead_kv=True, use_real_mot_mask=True`) instead of calling fvk
attention kernels directly — matching FlashRT's own established
production convention (the `AttentionBackendBase` protocol every other
real Thor pipeline in this codebase, e.g. Pi0.5's, uses uniformly) so
the pipeline-owned per-layer KV cache stays the single source of truth
an eventual real-checkpoint-loading frontend can plug into unchanged.

Double-stream block (real order, per side txt/img):
    x_normed = LayerNorm_no_affine(x)
    x_mod = (1+scale1)*x_normed + shift1
    q,k,v = x_mod @ {q,k,v}_weight              # separate GEMMs, real per-head width
    q,k = QKNorm(q,k)                            # rms_norm_fp16 with the real scale weight
    (RoPE applied once below, over the full combined [txt|img] sequence)
    attn = joint_attention(q,k,v)                # ImageWAMAttnBackend "backbone" site, no mask
    x = x + gate1 * (attn @ proj_weight)
    x_normed2 = LayerNorm_no_affine(x)
    x_mod2 = (1+scale2)*x_normed2 + shift2
    mlp = silu_glu(x_mod2 @ mlp0_weight) @ mlp2_weight
    x = x + gate2 * mlp

Single-stream block: same shape, one stream (no txt/img split), one
set of q/k/v/mlp_in weights, attn-out-proj + mlp-down SUMMED before the
one gated residual (matches the real fused `linear2`, represented here
as two separate GEMMs — see `real_single_stream_block.py`'s own
docstring for why that split is mathematically identical).

ActionDiT double/single blocks: same shape at `action_hidden_dim`/
`action_attn_width` (which differ, unlike the backbone), IMG-ONLY (no
txt branch), joint attention against the "mot" site (Q = action rows
only, K/V = the full combined sequence including the frozen backbone
K/V cache written during prefill — no mask, same real rule as
"backbone").

# AdaLN modulation is precomputed OUTSIDE this module, once
=======================================================================

Real AdaLN modulation depends on a per-FORWARD timestep, shared across
every layer of a given stream type — the backbone's is FIXED for the
whole forward (real inference always conditions the reference/context
encode on `timestep=0`, confirmed against the real checkpoint run
above), and ActionDiT's changes once per denoise STEP (but `step` is
already a compile-time Python constant during CUDA Graph capture — see
`imagewam_denoise_step`'s own docstring below), so this module never
recomputes AdaLN inside a per-replay-cost path. The caller (the
frontend, Phase 5) is expected to precompute the modulation tuples via
`flash_rt.models.imagewam.pipeline_real.compute_shared_modulation`/
`compute_action_modulation` ONCE before graph capture, and to
precompute the RoPE tables via
`flash_rt.models.imagewam.rope.build_backbone_rope_table`/
`build_action_rope_table` ONCE — then pass all of these in as the
`mod_txt`/`mod_img`/`mod_single`/`rope_table`/`action_mods`/
`action_rope_table` arguments below. Every one of these is captured by
the graph as a small, fixed-address read-only buffer; nothing here
allocates or recomputes them per replay.

# Weight pointer keys this file expects
=======================================================================

See `flash_rt/frontends/torch/_imagewam_thor_spec.py` for the full
per-layer shape declarations this file's weight dict keys must match
(`weights[("backbone", stream, layer, slot)]` / `weights[("action_dit",
stream, layer, slot)]`, `slot` matching that spec file's own suffix
names). K/V weights are real per-head width (`hidden`/`action_attn_width`,
NOT the old broadcast `HD` width); `*_query_norm`/`*_key_norm` are new
QK-Norm scale weights; `*mlp0`/`mlp_in` widths are `mlp_hidden*2` (real
SiLU-gated GLU).

`bufs` keys (pipeline-owned scratch, pre-allocated once by the
frontend, fp16 throughout unless noted):
    context           (max_txt_seq, joint_attention_dim)  -- input
    backbone_hidden   (a0, hidden)   -- the persistent residual
    normed_scratch    (a0, hidden)   -- LayerNorm output, reused by
                                         every sub-block
    modded_scratch    (a0, hidden)   -- post-AdaLN-modulation, reused
                                         by every sub-block
    txt_mlp_merged/txt_mlp_gated, img_mlp_merged/img_mlp_gated
                      (x0 or img_len, mlp_hidden*2 / mlp_hidden)
    single_mlp_merged/single_mlp_gated  (a0, mlp_hidden*2 / mlp_hidden)
    proj_scratch      (a0, hidden)   -- GEMM output landing pad before
                                         the gated-residual accumulate
    proj_scratch2     (a0, hidden)   -- single-stream block's own
                                         second landing pad (attn-out
                                         and mlp-down are summed before
                                         ONE gated residual)
    action_hidden     (num_action, action_hidden_dim)
    action_normed / action_modded          (num_action, action_hidden_dim)
    action_proj_scratch                    (num_action, action_hidden_dim)
    action_mlp_merged/action_mlp_gated     (num_action, action_mlp_hidden*2 / action_mlp_hidden)
    action_latent     (num_action, action_hidden_dim), F32 -- the
                       running flow-matching state
"""
from __future__ import annotations

import torch


def _ptr_offset(base_ptr: int, row_offset: int, row_width: int) -> int:
    """Byte offset into a row-major fp16 buffer -- 2 bytes/element."""
    return int(base_ptr) + int(row_offset) * int(row_width) * 2


def _wrap_fp16(ptr: int, seq: int, dim: int) -> torch.Tensor:
    """Zero-copy CUDA tensor view over a raw fp16 pointer -- same
    technique as `flash_rt.hardware.thor.attn_backend._fp16_tensor_from_ptr`
    (not shared via import to keep this module's only external
    dependency `torch`, matching every other pointer-based pipeline
    file in this project). Used only for the small AdaLN elementwise
    steps below (modulate / gated-residual / plain add) -- every GEMM
    and every fvk kernel call still operates on raw pointers directly.
    """
    interface = {
        "data": (int(ptr), False),
        "shape": (int(seq), int(dim)),
        "typestr": "<f2",
        "version": 3,
    }
    owner = type("_Fp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device="cuda")


def _modulate(normed_ptr: int, out_ptr: int, shift, scale, seq: int, dim: int) -> None:
    """Real `(1+scale)*normed + shift`, writing into `out_ptr` (may
    equal `normed_ptr`). `shift`/`scale` are `(1,1,dim)` float32
    tensors from `adaln.modulation`'s own chunk output -- see that
    module's docstring. All ops are in-place on existing buffers or
    tiny (dim-sized) allocations, both safe and cheap under CUDA Graph
    capture (PyTorch's graph-private memory pool handles allocation
    during capture correctly; only a raw, non-pooled `cudaMalloc` --
    e.g. `GemmRunner`'s C++ constructor -- breaks capture, not this).
    """
    normed = _wrap_fp16(normed_ptr, seq, dim)
    out = _wrap_fp16(out_ptr, seq, dim)
    out.copy_(normed)
    out.mul_(1.0 + scale[0].to(out.dtype))
    out.add_(shift[0].to(out.dtype))


def _gated_residual_inplace(residual_ptr: int, sublayer_ptr: int, gate, seq: int, dim: int) -> None:
    """Real `residual += gate * sublayer_out`, in place at `residual_ptr`."""
    residual = _wrap_fp16(residual_ptr, seq, dim)
    sublayer = _wrap_fp16(sublayer_ptr, seq, dim)
    residual.add_(gate[0].to(residual.dtype) * sublayer)


def _add_inplace(dst_ptr: int, src_ptr: int, seq: int, dim: int) -> None:
    """Plain `dst += src`, in place -- used to sum the single-stream
    block's attn-out and mlp-down projections before their ONE shared
    gated residual (see module docstring)."""
    dst = _wrap_fp16(dst_ptr, seq, dim)
    src = _wrap_fp16(src_ptr, seq, dim)
    dst.add_(src)


def _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn,
                          mod_txt, mod_img, rope_table):
    """One real FLUX.2 double-stream block: separate img/txt Q/K/V+MLP,
    joint attn. Reads/writes `bufs["backbone_hidden"]` rows [0,x0)
    (text) and [x0,a0) (image) in place.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    NH = dims["NH"]
    mlp_hidden = dims["mlp_hidden"]
    joint_attention_dim = dims["joint_attention_dim"]
    x0 = dims["x0"]
    a0 = dims["a0"]
    img_len = a0 - x0
    eps = 1e-6
    key = lambda slot: weights[("backbone", "double", layer_idx, slot)]

    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = mod_txt
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = mod_img

    combined = bufs["backbone_hidden"]  # (a0, hidden)
    normed = bufs["normed_scratch"]
    modded = bufs["modded_scratch"]

    ptrs = attn.get_slot_ptrs("backbone", layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]  # real per-head width (hidden)

    # --- text stream: freshly re-derived from raw context (kept
    # simplification, see the git history of this file / plan.md),
    # overwrites rows [0,x0) of combined ---
    gemm.fp16_nn(bufs["context"], key("txt_in.weight"), combined, x0, hidden, joint_attention_dim, stream)
    txt_x = combined  # rows [0, x0)

    fvk.layer_norm_no_affine_fp16(txt_x, normed, x0, hidden, eps, stream)
    _modulate(normed, modded, txt_shift1, txt_scale1, x0, hidden)
    gemm.fp16_nn(modded, key("txt_q.weight"), Q_O, x0, hidden, hidden, stream)
    gemm.fp16_nn(modded, key("txt_k.weight"), K_cache, x0, hidden, hidden, stream)
    gemm.fp16_nn(modded, key("txt_v.weight"), V_cache, x0, hidden, hidden, stream)
    fvk.rms_norm_fp16(Q_O, key("txt_query_norm"), Q_O, x0 * NH, HD, eps, stream)
    fvk.rms_norm_fp16(K_cache, key("txt_key_norm"), K_cache, x0 * NH, HD, eps, stream)

    # --- image stream: persistent residual, rows [x0,a0) of combined ---
    img_x_ptr = _ptr_offset(combined, x0, hidden)
    img_normed_ptr = _ptr_offset(normed, x0, hidden)
    img_modded_ptr = _ptr_offset(modded, x0, hidden)
    img_Q_ptr = _ptr_offset(Q_O, x0, hidden)
    img_K_ptr = _ptr_offset(K_cache, x0, hidden)
    img_V_ptr = _ptr_offset(V_cache, x0, hidden)

    fvk.layer_norm_no_affine_fp16(img_x_ptr, img_normed_ptr, img_len, hidden, eps, stream)
    _modulate(img_normed_ptr, img_modded_ptr, img_shift1, img_scale1, img_len, hidden)
    gemm.fp16_nn(img_modded_ptr, key("img_q.weight"), img_Q_ptr, img_len, hidden, hidden, stream)
    gemm.fp16_nn(img_modded_ptr, key("img_k.weight"), img_K_ptr, img_len, hidden, hidden, stream)
    gemm.fp16_nn(img_modded_ptr, key("img_v.weight"), img_V_ptr, img_len, hidden, hidden, stream)
    fvk.rms_norm_fp16(img_Q_ptr, key("img_query_norm"), img_Q_ptr, img_len * NH, HD, eps, stream)
    fvk.rms_norm_fp16(img_K_ptr, key("img_key_norm"), img_K_ptr, img_len * NH, HD, eps, stream)

    # --- RoPE over the FULL combined [txt|img] sequence, once each
    # for Q and K (matches real_double_stream_block_forward_fp16) ---
    fvk.rope_apply_fp16_perhead(Q_O, rope_table, a0, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(K_cache, rope_table, a0, NH, HD, stream)

    # --- joint self-attention over the whole [text | image] sequence,
    # real per-head, no mask (opportunities.md OPT-002's correction) ---
    attn.run("backbone", layer_idx, q_seq=a0, stream=stream)

    # --- separate output projections, GATED residual ---
    proj = bufs["proj_scratch"]
    gemm.fp16_nn(Q_O, key("txt_proj.weight"), proj, x0, hidden, hidden, stream)
    _gated_residual_inplace(txt_x, proj, txt_gate1, x0, hidden)

    img_proj_ptr = _ptr_offset(proj, x0, hidden)
    gemm.fp16_nn(img_Q_ptr, key("img_proj.weight"), img_proj_ptr, img_len, hidden, hidden, stream)
    _gated_residual_inplace(img_x_ptr, img_proj_ptr, img_gate1, img_len, hidden)

    # --- separate real SiLU-GLU MLPs, GATED residual ---
    fvk.layer_norm_no_affine_fp16(txt_x, normed, x0, hidden, eps, stream)
    _modulate(normed, modded, txt_shift2, txt_scale2, x0, hidden)
    txt_mlp_merged, txt_mlp_gated = bufs["txt_mlp_merged"], bufs["txt_mlp_gated"]
    gemm.fp16_nn(modded, key("txt_mlp0.weight"), txt_mlp_merged, x0, mlp_hidden * 2, hidden, stream)
    fvk.silu_glu_merged_fp16(txt_mlp_merged, txt_mlp_gated, x0, mlp_hidden, stream)
    gemm.fp16_nn(txt_mlp_gated, key("txt_mlp2.weight"), proj, x0, hidden, mlp_hidden, stream)
    _gated_residual_inplace(txt_x, proj, txt_gate2, x0, hidden)

    fvk.layer_norm_no_affine_fp16(img_x_ptr, img_normed_ptr, img_len, hidden, eps, stream)
    _modulate(img_normed_ptr, img_modded_ptr, img_shift2, img_scale2, img_len, hidden)
    img_mlp_merged, img_mlp_gated = bufs["img_mlp_merged"], bufs["img_mlp_gated"]
    gemm.fp16_nn(img_modded_ptr, key("img_mlp0.weight"), img_mlp_merged, img_len, mlp_hidden * 2, hidden, stream)
    fvk.silu_glu_merged_fp16(img_mlp_merged, img_mlp_gated, img_len, mlp_hidden, stream)
    gemm.fp16_nn(img_mlp_gated, key("img_mlp2.weight"), img_proj_ptr, img_len, hidden, mlp_hidden, stream)
    _gated_residual_inplace(img_x_ptr, img_proj_ptr, img_gate2, img_len, hidden)


def _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn, mod_single, rope_table):
    """One real FLUX.2 single-stream block: merged img+txt, separate
    q/k/v/mlp_in GEMMs (mathematically identical to the real fused
    `linear1`, see `real_single_stream_block.py`'s own docstring).
    Operates on the whole `bufs["backbone_hidden"]` (a0, hidden)
    buffer. ``weight_layer_idx`` (0..19) indexes this stream's own
    declared weights; ``site_layer_idx`` continues after the
    double-stream layers, indexing the "backbone" attention site's
    shared 25-layer KV cache.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    NH = dims["NH"]
    mlp_hidden = dims["mlp_hidden"]
    a0 = dims["a0"]
    eps = 1e-6
    key = lambda slot: weights[("backbone", "single", weight_layer_idx, slot)]
    shift, scale, gate = mod_single

    combined = bufs["backbone_hidden"]
    normed = bufs["normed_scratch"]
    modded = bufs["modded_scratch"]

    ptrs = attn.get_slot_ptrs("backbone", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

    fvk.layer_norm_no_affine_fp16(combined, normed, a0, hidden, eps, stream)
    _modulate(normed, modded, shift, scale, a0, hidden)

    gemm.fp16_nn(modded, key("q.weight"), Q_O, a0, hidden, hidden, stream)
    gemm.fp16_nn(modded, key("k.weight"), K_cache, a0, hidden, hidden, stream)
    gemm.fp16_nn(modded, key("v.weight"), V_cache, a0, hidden, hidden, stream)
    fvk.rms_norm_fp16(Q_O, key("query_norm"), Q_O, a0 * NH, HD, eps, stream)
    fvk.rms_norm_fp16(K_cache, key("key_norm"), K_cache, a0 * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(Q_O, rope_table, a0, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(K_cache, rope_table, a0, NH, HD, stream)

    mlp_merged, mlp_gated = bufs["single_mlp_merged"], bufs["single_mlp_gated"]
    gemm.fp16_nn(modded, key("mlp_in.weight"), mlp_merged, a0, mlp_hidden * 2, hidden, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, a0, mlp_hidden, stream)

    attn.run("backbone", site_layer_idx, q_seq=a0, stream=stream)

    from_attn = bufs["proj_scratch"]
    from_mlp = bufs["proj_scratch2"]
    gemm.fp16_nn(Q_O, key("attn_out_proj.weight"), from_attn, a0, hidden, hidden, stream)
    gemm.fp16_nn(mlp_gated, key("mlp_down.weight"), from_mlp, a0, hidden, mlp_hidden, stream)
    _add_inplace(from_attn, from_mlp, a0, hidden)
    _gated_residual_inplace(combined, from_attn, gate, a0, hidden)


def imagewam_encode_once(ctx, fvk, gemm, bufs, weights, dims, stream=0):
    """Target-image encode step -- explicit no-op in this scope.

    ImageWAM's real encode step is a VAE forward
    (`_encode_flux2_image_tokens`); no VAE weights are declared (out of
    scope -- see `_imagewam_thor_spec.py` and `opportunities.md`
    OPT-001/OPT-008). The frontend fills `bufs["backbone_hidden"]`'s
    image rows `[x0, a0)` directly, standing in for already-encoded
    image patch tokens, so there is nothing left for this function to
    compute. Kept as its own pipeline-stage function (matching the
    Interface in plan.md) so a real VAE integration has an unambiguous
    place to go later, rather than folding "encode" into "prefill"
    silently.
    """
    return


def imagewam_prefill(ctx, fvk, gemm, bufs, weights, dims, stream=0, *, attn=None,
                      mod_txt=None, mod_img=None, mod_single=None, rope_table=None):
    """One real backbone forward (5 double-stream + 20 single-stream
    layers) over the [prefix | target-image] sequence, populating the
    per-layer KV cache the later denoise loop reads through the "mot"
    attention site.

    Required `dims` keys: hidden, HD, NH, mlp_hidden, joint_attention_dim,
    x0, a0, num_layers_double, num_layers_single.

    `mod_txt`/`mod_img`/`mod_single`/`rope_table`: precomputed ONCE by
    the caller before graph capture -- see module docstring's "AdaLN
    modulation is precomputed outside this module" section
    (`flash_rt.models.imagewam.pipeline_real.compute_shared_modulation`
    / `flash_rt.models.imagewam.rope.build_backbone_rope_table`).

    `gemm` must be a `fvk.GemmRunner` constructed once by the caller
    outside any CUDA-graph-capturable region -- `GemmRunner()` does a
    raw `cudaMalloc` for a 256MB workspace at construction, which is
    NOT safe to call from inside a capturing stream (unlike ordinary
    torch tensor allocations, which DO go through the graph-safe
    caching allocator -- see `_modulate`'s own docstring). Confirmed by
    an earlier draft's real `cublasLtMatmul` internal error at runtime,
    not by inspection.
    """
    if attn is None:
        raise ValueError("imagewam_prefill requires an ImageWAMAttnBackend via attn=")
    if mod_txt is None or mod_img is None or mod_single is None or rope_table is None:
        raise ValueError(
            "imagewam_prefill requires mod_txt/mod_img/mod_single/rope_table -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_shared_modulation / "
            "flash_rt.models.imagewam.rope.build_backbone_rope_table")
    num_double = dims["num_layers_double"]
    for layer_idx in range(num_double):
        _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn,
                              mod_txt, mod_img, rope_table)
    # The "backbone" attention site's per-layer KV cache is ONE
    # contiguous 25-layer range (num_layers_double + num_layers_single);
    # single-stream layers continue that same indexing rather than
    # restarting at 0, which would otherwise alias double-stream layer
    # 0..4's own K/V cache slots.
    for i in range(dims["num_layers_single"]):
        _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn,
                              mod_single, rope_table)


# ──────────────────────────────────────────────────────────────────
# ActionDiT (denoise loop): joint attention against the backbone's
# frozen K/V cache via the "mot" site, flow-matching Euler update.
# ──────────────────────────────────────────────────────────────────
#
# ActionDiT is IMG-ONLY (no separate txt branch, see
# `real_action_expert.py`) -- its "double"/"single" naming mirrors the
# backbone's only so every backbone layer has a same-indexed ActionDiT
# layer for `mot_joint` validity.
#
# Per-step attention: only the action rows have a live query this
# step -- the backbone/image rows' own Q was already consumed during
# prefill and is never read again (OPT-003, opportunities.md).
# `attn.run("mot", ..., q_seq=num_action, kv_seq=total, ...)` computes
# attention for ONLY the action rows; K/V still cover the whole
# combined sequence AND include no mask (opportunities.md's OPT-002/
# OPT-003 "Major correction" -- action sees the whole [text|ref|action]
# sequence, confirmed against the real checkpoint). Rows `[a0, total)`
# of the shared Q_O/K_cache/V_cache are overwritten with this step's
# fresh ActionDiT Q/K/V before every `attn.run` call; rows `[0, a0)`
# are left exactly as prefill last wrote them.
#
# ActionDiT has no declared output-projection head to a real small
# action_dim (no real `action_encoder`/output head modeled here -- see
# opportunities.md's `imagewam_full_forward_real` docstring for the
# same documented placeholder) -- this pipeline treats the ActionDiT's
# own final hidden state as the velocity directly, same width as
# `bufs["action_latent"]`.


def _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, site_layer_idx, stream, attn,
                          mod, action_rope_table):
    action_hidden_dim = dims["action_hidden_dim"]
    action_attn_width = dims["action_attn_width"]  # == backbone's `hidden`, required for mot_joint
    HD = dims["HD"]
    NH = dims["NH"]
    action_mlp_hidden = dims["action_mlp_hidden"]
    x0 = dims["x0"]
    a0 = dims["a0"]
    num_action = dims["num_action"]
    eps = 1e-6
    key = lambda slot: weights[("action_dit", "double", layer_idx, slot)]
    (shift1, scale1, gate1), (shift2, scale2, gate2) = mod

    action_x = bufs["action_hidden"]  # (num_action, action_hidden_dim)
    normed = bufs["action_normed"]
    modded = bufs["action_modded"]

    ptrs = attn.get_slot_ptrs("mot", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
    # Q_O/K_cache/V_cache's row width is action_attn_width (shared with
    # the backbone, now real per-head width), NOT action_hidden_dim --
    # the residual stream and the attention-facing projection are
    # different widths for ActionDiT (unlike the backbone).
    action_Q_ptr = _ptr_offset(Q_O, a0, action_attn_width)
    action_K_ptr = _ptr_offset(K_cache, a0, action_attn_width)
    action_V_ptr = _ptr_offset(V_cache, a0, action_attn_width)

    fvk.layer_norm_no_affine_fp16(action_x, normed, num_action, action_hidden_dim, eps, stream)
    _modulate(normed, modded, shift1, scale1, num_action, action_hidden_dim)
    gemm.fp16_nn(modded, key("q.weight"), action_Q_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    gemm.fp16_nn(modded, key("k.weight"), action_K_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    gemm.fp16_nn(modded, key("v.weight"), action_V_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    fvk.rms_norm_fp16(action_Q_ptr, key("query_norm"), action_Q_ptr, num_action * NH, HD, eps, stream)
    fvk.rms_norm_fp16(action_K_ptr, key("key_norm"), action_K_ptr, num_action * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(action_Q_ptr, action_rope_table, num_action, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(action_K_ptr, action_rope_table, num_action, NH, HD, stream)

    attn.run("mot", site_layer_idx, q_seq=num_action, kv_seq=dims["total"], stream=stream, x0=x0, a0=a0)

    proj = bufs["action_proj_scratch"]
    gemm.fp16_nn(action_Q_ptr, key("proj.weight"), proj, num_action, action_hidden_dim, action_attn_width, stream)
    _gated_residual_inplace(action_x, proj, gate1, num_action, action_hidden_dim)

    fvk.layer_norm_no_affine_fp16(action_x, normed, num_action, action_hidden_dim, eps, stream)
    _modulate(normed, modded, shift2, scale2, num_action, action_hidden_dim)
    mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
    gemm.fp16_nn(modded, key("mlp0.weight"), mlp_merged, num_action, action_mlp_hidden * 2, action_hidden_dim, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)
    gemm.fp16_nn(mlp_gated, key("mlp2.weight"), proj, num_action, action_hidden_dim, action_mlp_hidden, stream)
    _gated_residual_inplace(action_x, proj, gate2, num_action, action_hidden_dim)


def _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn, mod, action_rope_table):
    action_hidden_dim = dims["action_hidden_dim"]
    action_attn_width = dims["action_attn_width"]
    HD = dims["HD"]
    NH = dims["NH"]
    action_mlp_hidden = dims["action_mlp_hidden"]
    x0 = dims["x0"]
    a0 = dims["a0"]
    num_action = dims["num_action"]
    eps = 1e-6
    key = lambda slot: weights[("action_dit", "single", weight_layer_idx, slot)]
    shift, scale, gate = mod

    action_x = bufs["action_hidden"]
    normed = bufs["action_normed"]
    modded = bufs["action_modded"]

    ptrs = attn.get_slot_ptrs("mot", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
    action_Q_ptr = _ptr_offset(Q_O, a0, action_attn_width)
    action_K_ptr = _ptr_offset(K_cache, a0, action_attn_width)
    action_V_ptr = _ptr_offset(V_cache, a0, action_attn_width)

    fvk.layer_norm_no_affine_fp16(action_x, normed, num_action, action_hidden_dim, eps, stream)
    _modulate(normed, modded, shift, scale, num_action, action_hidden_dim)
    gemm.fp16_nn(modded, key("q.weight"), action_Q_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    gemm.fp16_nn(modded, key("k.weight"), action_K_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    gemm.fp16_nn(modded, key("v.weight"), action_V_ptr, num_action, action_attn_width, action_hidden_dim, stream)
    fvk.rms_norm_fp16(action_Q_ptr, key("query_norm"), action_Q_ptr, num_action * NH, HD, eps, stream)
    fvk.rms_norm_fp16(action_K_ptr, key("key_norm"), action_K_ptr, num_action * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(action_Q_ptr, action_rope_table, num_action, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(action_K_ptr, action_rope_table, num_action, NH, HD, stream)

    attn.run("mot", site_layer_idx, q_seq=num_action, kv_seq=dims["total"], stream=stream, x0=x0, a0=a0)

    mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
    gemm.fp16_nn(modded, key("mlp_in.weight"), mlp_merged, num_action, action_mlp_hidden * 2, action_hidden_dim, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)

    from_attn = bufs["action_proj_scratch"]
    from_mlp = bufs["action_proj_scratch2"]
    gemm.fp16_nn(action_Q_ptr, key("attn_out_proj.weight"), from_attn, num_action, action_hidden_dim, action_attn_width, stream)
    gemm.fp16_nn(mlp_gated, key("mlp_down.weight"), from_mlp, num_action, action_hidden_dim, action_mlp_hidden, stream)
    _add_inplace(from_attn, from_mlp, num_action, action_hidden_dim)
    _gated_residual_inplace(action_x, from_attn, gate, num_action, action_hidden_dim)


def imagewam_denoise_step(ctx, fvk, gemm, bufs, weights, dims, step, stream=0, *, attn=None,
                           mod_double=None, mod_single=None, action_rope_table=None):
    """One flow-matching Euler step of the ActionDiT denoise loop.

    `step` is a plain Python int -- a compile-time constant during CUDA
    Graph capture (this project's own dt schedule is a fixed uniform
    `1.0 / num_denoise_steps`). `mod_double`/`mod_single` are THIS
    STEP's own precomputed AdaLN modulation (ActionDiT's conditioning
    timestep changes every step, but since `step` is itself a
    compile-time constant, so is every step's timestep -- the caller
    precomputes one modulation tuple PER STEP before capture, see
    module docstring, and `imagewam_denoise_loop` below selects the
    right one per iteration).

    Required `dims` keys (beyond `imagewam_prefill`'s): action_hidden_dim,
    action_attn_width (must equal `hidden`), HD, action_mlp_hidden, x0, a0,
    total, num_action, action_num_layers_double, action_num_layers_single, dt.
    """
    if attn is None:
        raise ValueError("imagewam_denoise_step requires an ImageWAMAttnBackend via attn=")
    if mod_double is None or mod_single is None or action_rope_table is None:
        raise ValueError(
            "imagewam_denoise_step requires mod_double/mod_single/action_rope_table -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_action_modulation / "
            "flash_rt.models.imagewam.rope.build_action_rope_table")
    num_action = dims["num_action"]
    action_hidden_dim = dims["action_hidden_dim"]
    n = num_action * action_hidden_dim

    fvk.gpu_cast_fp32_to_fp16(bufs["action_latent"], bufs["action_hidden"], n, stream)

    num_double = dims["action_num_layers_double"]
    for layer_idx in range(num_double):
        _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, layer_idx, stream, attn,
                              mod_double, action_rope_table)
    for i in range(dims["action_num_layers_single"]):
        _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn,
                              mod_single, action_rope_table)

    fvk.gpu_euler_step(bufs["action_latent"], bufs["action_hidden"],
                        num_action, action_hidden_dim, dims["dt"], 0, stream)


def imagewam_denoise_loop(ctx, fvk, gemm, bufs, weights, dims, stream=0, *, attn=None,
                           action_mods=None, action_rope_table=None):
    """The whole flow-matching denoise loop -- what the frontend
    captures as ONE CUDA Graph together with `imagewam_prefill`.

    `action_mods`: a list of `(mod_double, mod_single)` tuples, one per
    denoise step (`dims["num_denoise_steps"]` entries), precomputed
    ONCE by the caller before capture -- see module docstring.
    """
    if action_mods is None or action_rope_table is None:
        raise ValueError(
            "imagewam_denoise_loop requires action_mods/action_rope_table -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_action_modulation")
    for step in range(dims["num_denoise_steps"]):
        mod_double, mod_single = action_mods[step]
        imagewam_denoise_step(ctx, fvk, gemm, bufs, weights, dims, step, stream=stream, attn=attn,
                               mod_double=mod_double, mod_single=mod_single,
                               action_rope_table=action_rope_table)

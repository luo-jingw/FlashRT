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
    x_mod = ada_layer_norm(x, scale1, shift1)     # ONE fused kernel (OPT-004 step 3): LN_no_affine(x)*(1+scale1)+shift1
    qkv = x_mod @ qkv_weight                     # ONE fused GEMM (OPT-004 step 2), real per-head width
    q,k,v = split_columns(qkv)                    # 3 slice-copies into Q_O/K_cache/V_cache
    q,k = QKNorm(q,k)                             # rms_norm_fp16 with the real scale weight
    (RoPE applied once below, over the full combined [txt|img] sequence)
    attn = joint_attention(q,k,v)                # ImageWAMAttnBackend "backbone" site, no mask
    x += gate1 * (attn @ proj_weight)             # ONE fused kernel (OPT-004 step 3): gate_res_fp16
    x_mod2 = ada_layer_norm(x, scale2, shift2)
    mlp = silu_glu(x_mod2 @ mlp0_weight) @ mlp2_weight
    x += gate2 * mlp

Single-stream block: same shape, one stream (no txt/img split), one
fused `qkv` GEMM + one `mlp_in` GEMM (both real fused `linear1` slices,
see `real_single_stream_block.py`'s own docstring), attn-out-proj +
mlp-down SUMMED before the one gated residual.

**QKV fusion (OPT-004 step 2, 2026-09-14)**: `qkv` is ONE GEMM into a
`(seq, 3*width)` scratch buffer (matches a real checkpoint's own fused
tensor directly — see `_imagewam_thor_spec.py`'s docstring), then
`_copy_slice` lands each third into its own real destination (`Q_O`,
`K_cache`, `V_cache` are three DIFFERENT persistent buffers the
attention backend owns, so the split can't be avoided entirely — it
moves from 3 GEMM launches to 1 GEMM + 3 contiguous copies). `proj`/
`attn_out_proj`/`mlp0`/`mlp2` (or `mlp_in`/`mlp_down`) remain separate
GEMMs; only the QKV *input* projection fuses.

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

# Weight dict values are CALLABLE linear ops, not raw pointers (OPT-004 step 5)
=======================================================================

See `flash_rt/frontends/torch/_imagewam_thor_spec.py` for the full
per-layer shape declarations this file's weight dict keys must match
(`weights[("backbone", stream, layer, slot)]` / `weights[("action_dit",
stream, layer, slot)]`, `slot` matching that spec file's own suffix
names). K/V weights are real per-head width (`hidden`/`action_attn_width`,
NOT the old broadcast `HD` width); `*_query_norm`/`*_key_norm` are new
QK-Norm scale weights; `*mlp0`/`mlp_in` widths are `mlp_hidden*2` (real
SiLU-gated GLU).

**`weights[key]` is a CALLABLE (`flash_rt.models.imagewam.quant_linear.Fp16Linear`/
`Fp8Linear`/`Nvfp4Linear`), not a raw pointer int** (2026-09-14,
`plan.md`'s "OPT-004 step 5" plan) — every weight-PROJECTION GEMM call
site in this file is `key(slot)(x_ptr, out_ptr, m, stream)`, uniformly,
regardless of precision; no branching on precision anywhere in this
file. `imagewam_thor.py`'s own `_alloc_random_weights` constructs the
selected linear-op class ONCE per weight (wrapping/quantizing the real
weight there), based on its own `precision=` constructor parameter.
Only weight-projection GEMMs go through this indirection — QK-Norm,
RoPE, attention, `silu_glu_merged_fp16`, `ada_layer_norm_fp16`,
`gate_res_fp16`, and the QKV-slice `_copy_slice` calls are UNCHANGED,
still direct `fvk`/pointer calls (none of them are a GEMM against a
learned weight matrix, so precision doesn't apply to them).

**AdaLN modulation + gated residual now use FlashRT's own existing
fused kernels (OPT-004 step 3, 2026-09-14)**: `fvk.ada_layer_norm_fp16`
(`out = LayerNorm_no_affine(x)*(1+scale)+shift` in ONE kernel launch --
found already implemented for a different model, GROOT N1.6's own
DiT, and reused as-is here since the math is IDENTICAL) replaces the
earlier two-step `layer_norm_no_affine_fp16` + a separate torch
elementwise modulate; `fvk.gate_res_fp16` (`residual[i] +=
gemm_out[i]*gate[i]`, flat) replaces the earlier torch
broadcast-multiply-add. Both kernels need `shift`/`scale` as
contiguous fp16 `(dim,)` and `gate` as a contiguous fp16 `(seq,dim)`
BROADCAST-MATERIALIZED copy (`gate_res_fp16`'s own flat indexing has
no stride concept, unlike the zero-copy views used elsewhere in this
file) -- `_fuse_mod_group` below does this once per call site, cheaply
(a `(dim,)` cast + one real `(seq,dim)` copy), and — same reasoning as
`_modulate`'s own former docstring — this only ever runs during graph
capture/warmup, NEVER during `.replay()` (the whole point of capturing
a graph is that replay re-executes the recorded kernel launches
directly, without re-running any Python).

`bufs` keys (pipeline-owned scratch, pre-allocated once by the
frontend, fp16 throughout unless noted):
    context           (max_txt_seq, joint_attention_dim)  -- BF16 (real
                                         Qwen3-4B text conditioning can
                                         legitimately reach ~120000 in
                                         magnitude once fed through the
                                         real backbone's own trained
                                         weights at real x0=512 --
                                         opportunities.md OPT-001 "FP16
                                         residual overflow" -- FP16's
                                         ~65504 ceiling cannot hold
                                         that; BF16 has FP32's exponent
                                         range at the same 2 bytes/elem)
    backbone_hidden   (a0, hidden)   -- BF16, the persistent residual,
                                         same reason as `context` above
                                         (this is what `context`/
                                         `img_raw` get projected INTO
                                         via `txt_in`/`img_in`, so it
                                         inherits the same range need)
    modded_scratch    (a0, hidden)   -- FP16 (unaffected): every
                                         `ada_layer_norm_bf16in_fp16out`
                                         call re-normalizes the wide-
                                         range residual back to O(1-10)
                                         before writing here, reused by
                                         every sub-block
    txt_qkv_merged/img_qkv_merged  (x0 or img_len, 3*hidden)  -- fused
                      Q/K/V GEMM scratch (OPT-004 step 2), sliced by
                      `_copy_slice` into Q_O/K_cache/V_cache
    single_qkv_merged (a0, 3*hidden)  -- same, single-stream block
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
    action_modded                          (num_action, action_hidden_dim)
    action_qkv_merged (num_action, 3*action_attn_width)  -- fused
                      action Q/K/V GEMM scratch (double AND single
                      share this one buffer -- never live at once)
    action_proj_scratch                    (num_action, action_hidden_dim)
    action_mlp_merged/action_mlp_gated     (num_action, action_mlp_hidden*2 / action_mlp_hidden)
    action_latent     (num_action, action_dim), F32 -- the running
                       flow-matching state (OPT-001: real action_dim
                       width, e.g. 7 for LIBERO -- NOT action_hidden_dim;
                       see imagewam_denoise_step's own docstring for the
                       real action_encoder/head encode-decode wrapper
                       this now needs every step)
    action_latent_fp16 (num_action, action_dim), fp16 -- cast scratch
                       for action_encoder's own GEMM input
    velocity          (num_action, action_dim), fp16 -- head's own
                       GEMM output, this step's Euler velocity
    head_modded       (num_action, action_hidden_dim) -- head's own
                       AdaLN-modulated scratch before its final Linear
"""
from __future__ import annotations

import torch


def _ptr_offset(base_ptr: int, row_offset: int, row_width: int) -> int:
    """Byte offset into a row-major fp16 buffer -- 2 bytes/element."""
    return int(base_ptr) + int(row_offset) * int(row_width) * 2


def _wrap_fp16(ptr: int, seq: int, dim: int, row_stride: int | None = None) -> torch.Tensor:
    """Zero-copy CUDA tensor view over a raw fp16 pointer -- same
    technique as `flash_rt.hardware.thor.attn_backend._fp16_tensor_from_ptr`
    (not shared via import to keep this module's only external
    dependency `torch`, matching every other pointer-based pipeline
    file in this project). Used for the small AdaLN elementwise steps
    below (modulate / gated-residual / plain add) and for the QKV
    fusion's own column-slice copies (`_copy_slice`) -- every GEMM and
    every fvk kernel call still operates on raw pointers directly.

    `row_stride` (elements, not bytes): defaults to `dim` (a plain
    contiguous `(seq,dim)` view). Pass the WIDER buffer's own row
    width to view a narrower COLUMN SLICE of it across all `seq` rows
    (e.g. the Q third of a `(seq, 3*hidden)` fused-QKV scratch buffer,
    stride=`3*hidden`, width=`hidden`) without a copy.
    """
    stride = int(dim) if row_stride is None else int(row_stride)
    interface = {
        "data": (int(ptr), False),
        "shape": (int(seq), int(dim)),
        "strides": (stride * 2, 2),
        "typestr": "<f2",
        "version": 3,
    }
    owner = type("_Fp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device="cuda")


def _col_ptr(base_ptr: int, col_offset: int) -> int:
    """Byte offset to column `col_offset` within the SAME row-range as
    `base_ptr` -- 2 bytes/element. Companion to `_ptr_offset` (which
    advances by whole ROWS); this advances within one row's columns,
    for slicing a fused-QKV scratch buffer's Q/K/V thirds."""
    return int(base_ptr) + int(col_offset) * 2


def _copy_slice(dst_ptr: int, src_ptr: int, seq: int, dim: int, *,
                 dst_row_stride: int | None = None, src_row_stride: int | None = None) -> None:
    """Plain `dst[:] = src[:]` (no elementwise math), both viewed as
    `(seq, dim)` -- used to land one Q/K/V third of a fused-QKV GEMM's
    wider scratch output into its own real per-head-width destination
    buffer (`Q_O`/`K_cache`/`V_cache`), which `attn.run()`'s attention
    backend needs at ITS OWN row width, not interleaved with the other
    two thirds (opportunities.md OPT-004 step 2 -- QKV fusion)."""
    dst = _wrap_fp16(dst_ptr, seq, dim, row_stride=dst_row_stride)
    src = _wrap_fp16(src_ptr, seq, dim, row_stride=src_row_stride)
    dst.copy_(src)


def _fuse_mod_group(shift, scale, gate, seq: int, dim: int):
    """Precompute one (shift,scale,gate) AdaLN triple into the exact
    form `fvk.ada_layer_norm_fp16`/`fvk.gate_res_fp16` consume directly
    (OPT-004 step 3): `shift`/`scale` as contiguous fp16 `(dim,)`
    tensors, `gate` broadcast-MATERIALIZED into a contiguous fp16
    `(seq,dim)` tensor (`gate_res_fp16`'s own flat elementwise indexing
    has no stride concept, unlike the zero-copy `_wrap_fp16` views used
    elsewhere in this file -- one real copy is unavoidable here, but
    cheap: a `(dim,)`-sized cast plus one `(seq,dim)` materialize).
    `shift`/`scale`/`gate` are `(1,1,dim)` float32 tensors from
    `adaln.modulation`'s own chunk output.

    Returns the three TENSOR objects (not bare pointers) -- the caller
    MUST keep them referenced as local variables for as long as any
    `fvk` call still needs to read their `.data_ptr()` (the same
    dangling-pointer trap documented in `tests/test_imagewam_prefill.py`:
    a bare `.data_ptr()` int with no surviving tensor reference is a
    ticking time bomb, since PyTorch's caching allocator is free to
    hand that exact memory to the next allocation on the same stream
    the instant the tensor's refcount hits zero).

    Only ever runs during graph capture/warmup, never during
    `.replay()` -- capturing a CUDA Graph records the kernel launches
    this function's own OUTPUT tensors get read by; replay re-executes
    those recorded launches directly without any Python running again,
    so this per-call Python-side cost (a handful of small tensor ops)
    is not part of the measured steady state.
    """
    shift_t = shift[0, 0].to(torch.float16).contiguous()
    scale_t = scale[0, 0].to(torch.float16).contiguous()
    gate_t = gate[0, 0].to(torch.float16).expand(seq, dim).contiguous()
    return shift_t, scale_t, gate_t


def _fuse_mod_pair(shift, scale):
    """Same as `_fuse_mod_group` but for a GATE-LESS AdaLN pair (OPT-001,
    `adaln.head_modulation`'s own output) -- no `(seq,dim)` broadcast
    materialize needed since there's no `gate_res_fp16` call downstream,
    just `ada_layer_norm_fp16`'s own internally-broadcast `(dim,)` scale/
    shift. Same dangling-pointer caller contract as `_fuse_mod_group`."""
    shift_t = shift[0, 0].to(torch.float16).contiguous()
    scale_t = scale[0, 0].to(torch.float16).contiguous()
    return shift_t, scale_t


def _add_inplace(dst_ptr: int, src_ptr: int, seq: int, dim: int) -> None:
    """Plain `dst += src`, in place -- used to sum the single-stream
    block's attn-out and mlp-down projections before their ONE shared
    gated residual (see module docstring)."""
    dst = _wrap_fp16(dst_ptr, seq, dim)
    src = _wrap_fp16(src_ptr, seq, dim)
    dst.add_(src)


def _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn,
                          mod_txt, mod_img, rope_table):
    """One real FLUX.2 double-stream block: separate img/txt fused-QKV
    GEMM + MLP, joint attn. Reads/writes `bufs["backbone_hidden"]` rows [0,x0)
    (text) and [x0,a0) (image) in place.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    NH = dims["NH"]
    mlp_hidden = dims["mlp_hidden"]
    x0 = dims["x0"]
    a0 = dims["a0"]
    img_len = a0 - x0
    eps = 1e-6
    key = lambda slot: weights[("backbone", "double", layer_idx, slot)]

    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = mod_txt
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = mod_img
    # OPT-004 step 3: fuse LN+modulate and gated-residual into existing
    # FlashRT kernels -- see _fuse_mod_group's own docstring for why
    # these tensors must stay referenced as locals through this whole
    # function (dangling-pointer safety), and why this Python-side cost
    # is graph-capture-only, never per-replay.
    txt_shift1_t, txt_scale1_t, txt_gate1_t = _fuse_mod_group(txt_shift1, txt_scale1, txt_gate1, x0, hidden)
    txt_shift2_t, txt_scale2_t, txt_gate2_t = _fuse_mod_group(txt_shift2, txt_scale2, txt_gate2, x0, hidden)
    img_shift1_t, img_scale1_t, img_gate1_t = _fuse_mod_group(img_shift1, img_scale1, img_gate1, img_len, hidden)
    img_shift2_t, img_scale2_t, img_gate2_t = _fuse_mod_group(img_shift2, img_scale2, img_gate2, img_len, hidden)

    combined = bufs["backbone_hidden"]  # (a0, hidden)
    modded = bufs["modded_scratch"]

    ptrs = attn.get_slot_ptrs("backbone", layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]  # real per-head width (hidden)

    # --- text stream: PERSISTENT residual, rows [0,x0) of combined.
    # `txt_in`/`img_in` are projected ONCE, by `imagewam_prefill`,
    # before this loop even starts -- NOT here, and NOT per layer.
    # (Bug found + fixed 2026-09-15, real Thor run against the official
    # bf16 reference: this function used to re-run `txt_in`/`img_in`
    # from RAW `context`/`img_raw` at the top of EVERY double-stream
    # layer, overwriting whatever the PREVIOUS layer's attn+MLP had
    # just written -- so only the LAST double-stream layer's own
    # one-shot transform of the raw input ever survived into the
    # single-stream layers, discarding 4 of the 5 real double-stream
    # layers' worth of depth for both streams. The real FLUX.2 model
    # (`third_party/flux2/src/flux2/model.py`: `img = self.img_in(x);
    # txt = self.txt_in(ctx)` BEFORE `for block in self.double_blocks`)
    # projects once and carries the SAME evolving residual through
    # every block, exactly like `img_x_ptr` here was already (correctly)
    # documented as doing -- this was never actually true for either
    # stream until this fix. Real-Thor cosine vs the official model
    # went from all=0.559/txt=0.548/img=0.907 (with the bug) to
    # verify-pending (this fix, not yet re-measured on Thor) --
    # opportunities.md OPT-001 has the full account. `checkpoint_loader.py`'s
    # own "txt_in/img_in shared across every double layer" finding
    # (ONE real tensor, not L copies) is what made the one-time-call
    # correct without any weight-loading change.)
    txt_x = combined  # rows [0, x0)

    # ada_layer_norm_bf16in_fp16out (not the plain _fp16 kernel): `txt_x`
    # aliases `combined`, the persistent BF16 residual buffer -- see
    # this function's own docstring and opportunities.md OPT-001 "FP16
    # residual overflow" for why FP16 cannot hold this buffer's values.
    fvk.ada_layer_norm_bf16in_fp16out(txt_x, txt_scale1_t.data_ptr(), txt_shift1_t.data_ptr(), modded, x0, hidden, eps, stream)
    txt_qkv_merged = bufs["txt_qkv_merged"]  # (x0, 3*hidden)
    key("txt_qkv.weight")(modded, txt_qkv_merged, x0, stream)
    _copy_slice(Q_O, txt_qkv_merged, x0, hidden, src_row_stride=3 * hidden)
    _copy_slice(K_cache, _col_ptr(txt_qkv_merged, hidden), x0, hidden, src_row_stride=3 * hidden)
    _copy_slice(V_cache, _col_ptr(txt_qkv_merged, 2 * hidden), x0, hidden, src_row_stride=3 * hidden)
    fvk.rms_norm_fp16(Q_O, key("txt_query_norm"), Q_O, x0 * NH, HD, eps, stream)
    fvk.rms_norm_fp16(K_cache, key("txt_key_norm"), K_cache, x0 * NH, HD, eps, stream)

    # --- image stream: PERSISTENT residual, rows [x0,a0) of combined.
    # `img_in` is projected ONCE, by `imagewam_prefill`, before this
    # loop -- see `txt_x`'s own comment above for the bug this fixes
    # (2026-09-15, opportunities.md OPT-001) and why this is correct
    # now that it's a one-time call, not a per-layer one.
    img_x_ptr = _ptr_offset(combined, x0, hidden)
    img_modded_ptr = _ptr_offset(modded, x0, hidden)
    img_Q_ptr = _ptr_offset(Q_O, x0, hidden)
    img_K_ptr = _ptr_offset(K_cache, x0, hidden)
    img_V_ptr = _ptr_offset(V_cache, x0, hidden)

    fvk.ada_layer_norm_bf16in_fp16out(img_x_ptr, img_scale1_t.data_ptr(), img_shift1_t.data_ptr(),
                             img_modded_ptr, img_len, hidden, eps, stream)
    img_qkv_merged = bufs["img_qkv_merged"]  # (img_len, 3*hidden)
    key("img_qkv.weight")(img_modded_ptr, img_qkv_merged, img_len, stream)
    _copy_slice(img_Q_ptr, img_qkv_merged, img_len, hidden, src_row_stride=3 * hidden)
    _copy_slice(img_K_ptr, _col_ptr(img_qkv_merged, hidden), img_len, hidden, src_row_stride=3 * hidden)
    _copy_slice(img_V_ptr, _col_ptr(img_qkv_merged, 2 * hidden), img_len, hidden, src_row_stride=3 * hidden)
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
    key("txt_proj.weight")(Q_O, proj, x0, stream)
    fvk.gate_res_bf16res(proj, txt_gate1_t.data_ptr(), txt_x, x0 * hidden, stream)

    img_proj_ptr = _ptr_offset(proj, x0, hidden)
    key("img_proj.weight")(img_Q_ptr, img_proj_ptr, img_len, stream)
    fvk.gate_res_bf16res(img_proj_ptr, img_gate1_t.data_ptr(), img_x_ptr, img_len * hidden, stream)

    # --- separate real SiLU-GLU MLPs, GATED residual ---
    fvk.ada_layer_norm_bf16in_fp16out(txt_x, txt_scale2_t.data_ptr(), txt_shift2_t.data_ptr(), modded, x0, hidden, eps, stream)
    txt_mlp_merged, txt_mlp_gated = bufs["txt_mlp_merged"], bufs["txt_mlp_gated"]
    key("txt_mlp0.weight")(modded, txt_mlp_merged, x0, stream)
    fvk.silu_glu_merged_fp16(txt_mlp_merged, txt_mlp_gated, x0, mlp_hidden, stream)
    key("txt_mlp2.weight")(txt_mlp_gated, proj, x0, stream)
    fvk.gate_res_bf16res(proj, txt_gate2_t.data_ptr(), txt_x, x0 * hidden, stream)

    fvk.ada_layer_norm_bf16in_fp16out(img_x_ptr, img_scale2_t.data_ptr(), img_shift2_t.data_ptr(),
                             img_modded_ptr, img_len, hidden, eps, stream)
    img_mlp_merged, img_mlp_gated = bufs["img_mlp_merged"], bufs["img_mlp_gated"]
    key("img_mlp0.weight")(img_modded_ptr, img_mlp_merged, img_len, stream)
    fvk.silu_glu_merged_fp16(img_mlp_merged, img_mlp_gated, img_len, mlp_hidden, stream)
    key("img_mlp2.weight")(img_mlp_gated, img_proj_ptr, img_len, stream)
    fvk.gate_res_bf16res(img_proj_ptr, img_gate2_t.data_ptr(), img_x_ptr, img_len * hidden, stream)


def _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn, mod_single, rope_table):
    """One real FLUX.2 single-stream block: merged img+txt, fused
    `qkv` GEMM + separate `mlp_in` GEMM (both real fused `linear1`
    slices, see `real_single_stream_block.py`'s own docstring).
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
    shift_t, scale_t, gate_t = _fuse_mod_group(shift, scale, gate, a0, hidden)

    combined = bufs["backbone_hidden"]
    modded = bufs["modded_scratch"]

    ptrs = attn.get_slot_ptrs("backbone", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

    fvk.ada_layer_norm_bf16in_fp16out(combined, scale_t.data_ptr(), shift_t.data_ptr(), modded, a0, hidden, eps, stream)

    qkv_merged = bufs["single_qkv_merged"]  # (a0, 3*hidden)
    key("qkv.weight")(modded, qkv_merged, a0, stream)
    _copy_slice(Q_O, qkv_merged, a0, hidden, src_row_stride=3 * hidden)
    _copy_slice(K_cache, _col_ptr(qkv_merged, hidden), a0, hidden, src_row_stride=3 * hidden)
    _copy_slice(V_cache, _col_ptr(qkv_merged, 2 * hidden), a0, hidden, src_row_stride=3 * hidden)
    fvk.rms_norm_fp16(Q_O, key("query_norm"), Q_O, a0 * NH, HD, eps, stream)
    fvk.rms_norm_fp16(K_cache, key("key_norm"), K_cache, a0 * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(Q_O, rope_table, a0, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(K_cache, rope_table, a0, NH, HD, stream)

    mlp_merged, mlp_gated = bufs["single_mlp_merged"], bufs["single_mlp_gated"]
    key("mlp_in.weight")(modded, mlp_merged, a0, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, a0, mlp_hidden, stream)

    attn.run("backbone", site_layer_idx, q_seq=a0, stream=stream)

    from_attn = bufs["proj_scratch"]
    from_mlp = bufs["proj_scratch2"]
    key("attn_out_proj.weight")(Q_O, from_attn, a0, stream)
    key("mlp_down.weight")(mlp_gated, from_mlp, a0, stream)
    _add_inplace(from_attn, from_mlp, a0, hidden)
    fvk.gate_res_bf16res(from_attn, gate_t.data_ptr(), combined, a0 * hidden, stream)


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

    Projects `txt_in`/`img_in` from raw `context`/`img_raw` ONCE, here,
    before any layer runs -- matches the real FLUX.2 model exactly
    (fixed 2026-09-15, opportunities.md OPT-001; `_double_stream_layer`
    used to redo this every layer, a real bug found via a real-Thor
    comparison against the official model, see that function's own
    docstring).

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

    # `txt_in`/`img_in` projected ONCE here, before any double-stream
    # layer runs -- matches the real FLUX.2 model exactly (`model.py`:
    # `img = self.img_in(x); txt = self.txt_in(ctx)` BEFORE
    # `for block in self.double_blocks`). Bug fix, 2026-09-15
    # (opportunities.md OPT-001): `_double_stream_layer` used to redo
    # this at the top of EVERY layer, discarding the previous layer's
    # entire output -- see that function's own docstring for the full
    # account. `txt_in.weight`/`img_in.weight` are the SAME real tensor
    # for every `layer_idx` (`checkpoint_loader.py`'s own "shared
    # across every double layer" finding), so `layer_idx=0` is correct
    # here regardless of `num_layers_double`.
    x0, a0 = dims["x0"], dims["a0"]
    hidden = dims["hidden"]
    img_len = a0 - x0
    combined = bufs["backbone_hidden"]
    weights[("backbone", "double", 0, "txt_in.weight")](bufs["context"], combined, x0, stream)
    weights[("backbone", "double", 0, "img_in.weight")](
        bufs["img_raw"], _ptr_offset(combined, x0, hidden), img_len, stream)

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
# OPT-001: real `action_encoder`/`head` now modeled -- see
# `imagewam_denoise_step`'s own docstring for the encode/decode wrapper
# around these per-layer block functions. `pipeline_real.py`'s
# `imagewam_full_forward_real` still uses the OLD action_hidden_dim-
# width placeholder this comment used to describe -- that's a SEPARATE
# wiring-test reference for that module's own scope (see this file's
# git history), not compared against this pointer-based path for that
# specific mechanism, and intentionally left as-is.


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
    shift1_t, scale1_t, gate1_t = _fuse_mod_group(shift1, scale1, gate1, num_action, action_hidden_dim)
    shift2_t, scale2_t, gate2_t = _fuse_mod_group(shift2, scale2, gate2, num_action, action_hidden_dim)

    action_x = bufs["action_hidden"]  # (num_action, action_hidden_dim)
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

    fvk.ada_layer_norm_fp16(action_x, scale1_t.data_ptr(), shift1_t.data_ptr(),
                             modded, num_action, action_hidden_dim, eps, stream)
    qkv_merged = bufs["action_qkv_merged"]  # (num_action, 3*action_attn_width)
    key("qkv.weight")(modded, qkv_merged, num_action, stream)
    _copy_slice(action_Q_ptr, qkv_merged, num_action, action_attn_width, src_row_stride=3 * action_attn_width)
    _copy_slice(action_K_ptr, _col_ptr(qkv_merged, action_attn_width), num_action, action_attn_width,
                src_row_stride=3 * action_attn_width)
    _copy_slice(action_V_ptr, _col_ptr(qkv_merged, 2 * action_attn_width), num_action, action_attn_width,
                src_row_stride=3 * action_attn_width)
    fvk.rms_norm_fp16(action_Q_ptr, key("query_norm"), action_Q_ptr, num_action * NH, HD, eps, stream)
    fvk.rms_norm_fp16(action_K_ptr, key("key_norm"), action_K_ptr, num_action * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(action_Q_ptr, action_rope_table, num_action, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(action_K_ptr, action_rope_table, num_action, NH, HD, stream)

    attn.run("mot", site_layer_idx, q_seq=num_action, kv_seq=dims["total"], stream=stream, x0=x0, a0=a0)

    proj = bufs["action_proj_scratch"]
    key("proj.weight")(action_Q_ptr, proj, num_action, stream)
    fvk.gate_res_fp16(proj, gate1_t.data_ptr(), action_x, num_action * action_hidden_dim, stream)

    fvk.ada_layer_norm_fp16(action_x, scale2_t.data_ptr(), shift2_t.data_ptr(),
                             modded, num_action, action_hidden_dim, eps, stream)
    mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
    key("mlp0.weight")(modded, mlp_merged, num_action, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)
    key("mlp2.weight")(mlp_gated, proj, num_action, stream)
    fvk.gate_res_fp16(proj, gate2_t.data_ptr(), action_x, num_action * action_hidden_dim, stream)


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
    shift_t, scale_t, gate_t = _fuse_mod_group(shift, scale, gate, num_action, action_hidden_dim)

    action_x = bufs["action_hidden"]
    modded = bufs["action_modded"]

    ptrs = attn.get_slot_ptrs("mot", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
    action_Q_ptr = _ptr_offset(Q_O, a0, action_attn_width)
    action_K_ptr = _ptr_offset(K_cache, a0, action_attn_width)
    action_V_ptr = _ptr_offset(V_cache, a0, action_attn_width)

    fvk.ada_layer_norm_fp16(action_x, scale_t.data_ptr(), shift_t.data_ptr(),
                             modded, num_action, action_hidden_dim, eps, stream)
    qkv_merged = bufs["action_qkv_merged"]  # (num_action, 3*action_attn_width)
    key("qkv.weight")(modded, qkv_merged, num_action, stream)
    _copy_slice(action_Q_ptr, qkv_merged, num_action, action_attn_width, src_row_stride=3 * action_attn_width)
    _copy_slice(action_K_ptr, _col_ptr(qkv_merged, action_attn_width), num_action, action_attn_width,
                src_row_stride=3 * action_attn_width)
    _copy_slice(action_V_ptr, _col_ptr(qkv_merged, 2 * action_attn_width), num_action, action_attn_width,
                src_row_stride=3 * action_attn_width)
    fvk.rms_norm_fp16(action_Q_ptr, key("query_norm"), action_Q_ptr, num_action * NH, HD, eps, stream)
    fvk.rms_norm_fp16(action_K_ptr, key("key_norm"), action_K_ptr, num_action * NH, HD, eps, stream)
    fvk.rope_apply_fp16_perhead(action_Q_ptr, action_rope_table, num_action, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(action_K_ptr, action_rope_table, num_action, NH, HD, stream)

    attn.run("mot", site_layer_idx, q_seq=num_action, kv_seq=dims["total"], stream=stream, x0=x0, a0=a0)

    mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
    key("mlp_in.weight")(modded, mlp_merged, num_action, stream)
    fvk.silu_glu_merged_fp16(mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)

    from_attn = bufs["action_proj_scratch"]
    from_mlp = bufs["action_proj_scratch2"]
    key("attn_out_proj.weight")(action_Q_ptr, from_attn, num_action, stream)
    key("mlp_down.weight")(mlp_gated, from_mlp, num_action, stream)
    _add_inplace(from_attn, from_mlp, num_action, action_hidden_dim)
    fvk.gate_res_fp16(from_attn, gate_t.data_ptr(), action_x, num_action * action_hidden_dim, stream)


def imagewam_denoise_step(ctx, fvk, gemm, bufs, weights, dims, step, stream=0, *, attn=None,
                           mod_double=None, mod_single=None, head_mod=None, action_rope_table=None,
                           delta=None):
    """One flow-matching Euler step of the ActionDiT denoise loop.

    `step` is a plain Python int -- a compile-time constant during CUDA
    Graph capture. `mod_double`/`mod_single`/`head_mod` are THIS STEP's
    own precomputed AdaLN modulation (ActionDiT's conditioning timestep
    changes every step, but since `step` is itself a compile-time
    constant, so is every step's timestep -- the caller precomputes one
    modulation tuple PER STEP before capture, see module docstring, and
    `imagewam_denoise_loop` below selects the right one per iteration).

    `delta`: this step's own real (non-uniform) Euler step size --
    matches the real `WanContinuousFlowMatchScheduler.step()` exactly
    (`sample + model_output * delta`, opportunities.md OPT-009's
    follow-up, `flash_rt.models.imagewam.scheduler.build_inference_schedule`).
    A plain Python float, precomputed ONCE per step before capture,
    same convention as `mod_double`/`head_mod`. Defaults to `None`,
    which falls back to `dims["dt"]` (this project's own ORIGINAL
    fixed-uniform-schedule simplification) -- every existing caller
    that never set up a real schedule is unaffected.

    OPT-001: real `action_encoder`/`head` wrapper around the per-layer
    blocks, ported from `imagewam/models/backbones/action_dit_flux2.py`'s
    own `pre_dit`/`post_dit` (read directly, confirmed against the real
    checkpoint's own `mixtures.action.{action_encoder,head}.*` keys):
    `latents_action` (this pipeline's `bufs["action_latent"]`) lives at
    real `action_dim` width (e.g. 7 for LIBERO), NOT `action_hidden_dim`
    -- `action_encoder` (a real Linear WITH bias, the only biased
    weight in this whole project) re-projects the CURRENT noisy state
    up to `action_hidden_dim` every step (`pre_dit`), the per-layer
    blocks below run entirely at `action_hidden_dim` width unchanged,
    and `head` (AdaLN, no gate, since it's a final layer not a residual
    block -- see `adaln.head_modulation`'s own docstring) projects the
    result back down to `action_dim` width as this step's velocity
    (`post_dit`) before the Euler update -- confirmed against
    `imagewam.py`'s own `infer_action_flux2`: `latents_action =
    scheduler.step(pred_action, ...)` integrates in `action_dim` space,
    not `action_hidden_dim` space.

    Required `dims` keys (beyond `imagewam_prefill`'s): action_dim,
    action_hidden_dim, action_attn_width (must equal `hidden`), HD,
    action_mlp_hidden, x0, a0, total, num_action,
    action_num_layers_double, action_num_layers_single, dt.
    """
    if attn is None:
        raise ValueError("imagewam_denoise_step requires an ImageWAMAttnBackend via attn=")
    if mod_double is None or mod_single is None or head_mod is None or action_rope_table is None:
        raise ValueError(
            "imagewam_denoise_step requires mod_double/mod_single/head_mod/action_rope_table -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_action_modulation / "
            "compute_action_head_modulation / "
            "flash_rt.models.imagewam.rope.build_action_rope_table")
    num_action = dims["num_action"]
    action_dim = dims["action_dim"]
    action_hidden_dim = dims["action_hidden_dim"]
    eps = 1e-6
    key = lambda slot: weights[("action_dit", "shared", 0, slot)]

    # --- encode: real action_latent (f32, action_dim) -> action_hidden
    # (fp16, action_hidden_dim) via action_encoder (the one biased
    # weight in this project) ---
    fvk.gpu_cast_fp32_to_fp16(bufs["action_latent"], bufs["action_latent_fp16"], num_action * action_dim, stream)
    key("action_encoder.weight")(bufs["action_latent_fp16"], bufs["action_hidden"], num_action, stream)
    fvk.add_bias_fp16(bufs["action_hidden"], key("action_encoder.bias"), num_action, action_hidden_dim, stream)

    num_double = dims["action_num_layers_double"]
    for layer_idx in range(num_double):
        _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, layer_idx, stream, attn,
                              mod_double, action_rope_table)
    for i in range(dims["action_num_layers_single"]):
        _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn,
                              mod_single, action_rope_table)

    # --- decode: real head (AdaLN, no gate, + Linear) -> velocity
    # (fp16, action_dim); Euler step in action_dim space, matching
    # imagewam.py's own infer_action_flux2 exactly ---
    head_shift_t, head_scale_t = _fuse_mod_pair(*head_mod)
    fvk.ada_layer_norm_fp16(bufs["action_hidden"], head_scale_t.data_ptr(), head_shift_t.data_ptr(),
                             bufs["head_modded"], num_action, action_hidden_dim, eps, stream)
    key("head.linear.weight")(bufs["head_modded"], bufs["velocity"], num_action, stream)

    step_delta = dims["dt"] if delta is None else delta
    fvk.gpu_euler_step(bufs["action_latent"], bufs["velocity"],
                        num_action, action_dim, step_delta, 0, stream)


def imagewam_denoise_loop(ctx, fvk, gemm, bufs, weights, dims, stream=0, *, attn=None,
                           action_mods=None, head_mods=None, action_rope_table=None, deltas=None):
    """The whole flow-matching denoise loop -- what the frontend
    captures as ONE CUDA Graph together with `imagewam_prefill`.

    `action_mods`: a list of `(mod_double, mod_single)` tuples;
    `head_mods`: a list of `(shift, scale)` tuples -- both one per
    denoise step (`dims["num_denoise_steps"]` entries), precomputed
    ONCE by the caller before capture -- see module docstring.

    `deltas`: optional list of per-step real Euler step sizes (real
    non-uniform schedule, opportunities.md OPT-009's follow-up) --
    `None` (default) falls back to `dims["dt"]` for every step, this
    project's original fixed-uniform-schedule simplification.
    """
    if head_mods is None:
        raise ValueError(
            "imagewam_denoise_loop requires head_mods -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_action_head_modulation")
    if action_mods is None or action_rope_table is None:
        raise ValueError(
            "imagewam_denoise_loop requires action_mods/action_rope_table -- see "
            "flash_rt.models.imagewam.pipeline_real.compute_action_modulation")
    for step in range(dims["num_denoise_steps"]):
        mod_double, mod_single = action_mods[step]
        imagewam_denoise_step(ctx, fvk, gemm, bufs, weights, dims, step, stream=stream, attn=attn,
                               mod_double=mod_double, mod_single=mod_single, head_mod=head_mods[step],
                               action_rope_table=action_rope_table,
                               delta=None if deltas is None else deltas[step])

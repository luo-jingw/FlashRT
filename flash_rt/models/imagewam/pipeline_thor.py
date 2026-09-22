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
mlp-down SUMMED before the one gated residual. With
`dims["merge_qkv_mlp"]` the real `linear1` runs as ONE GEMM
(opportunities.md OPT-015); with `dims["merge_linear2"]` the real
`linear2` also runs as ONE GEMM over `[attn_out | mlp_act]` (roadmap
item 4), exactly the official block's structure.

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

**Gated residual + next AdaLN in one kernel (roadmap item 3,
`dims["fuse_res_norm"]`)**: `fvk.gate_res_ada_layer_norm_{bf16res,fp16}`
updates the residual and writes the AdaLN output that follows it --
AdaLN2 inside a double block, and across block boundaries the next
block's AdaLN1 (double -> double), the single blocks' AdaLN (last double
-> first single, per side), the next single block's AdaLN, and for
ActionDiT the head's AdaLN into `head_modded`. It reads gate/scale/shift
as `(dim,)` FP32 vectors straight from the modulation output and rounds
them to FP16 in-kernel, so the result is bit-identical to the unfused
`gate_res_*` + `ada_layer_norm_*` pair and no `_fuse_mod_group` copies
are recorded per layer. `imagewam_prefill`/`imagewam_denoise_step`
build the chain (`AdaLNTarget`); the per-layer functions take
`input_normed`/`next_*`.

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
    single_linear2_in (a0, hidden + mlp_hidden)  -- merged `linear2`
                      GEMM input `[attn_out | mlp_act]` (roadmap item 4;
                      `action_linear2_in` is the ActionDiT counterpart,
                      `(num_action, action_attn_width + action_mlp_hidden)`)
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

from dataclasses import dataclass

import torch

from flash_rt.models.imagewam.awq import AwqScaledLinear
from flash_rt.models.imagewam.quant_linear import CutlassFp16SwiGluMlp, Nvfp4Linear, Nvfp4SwiGluMlp


def _mlp_gate_up(fvk, key, gate_slot: str, modded_ptr: int, merged_ptr: int, gated_ptr: int,
                  m: int, mlp_hidden: int, stream: int) -> None:
    """The real SwiGLU MLP's own gate/up half:
    `gated = SiLU(x @ W_gate) * (x @ W_up)`. Three equivalent paths,
    selected by WHICH CLASS `weights[key]` already is (constructed by
    `imagewam_thor.py`, this module stays precision-agnostic, same
    convention as every other `weights[key](...)` call site):

    - Default (`Fp16Linear`/`Fp8Linear`/etc.): ONE merged GEMM into
      `merged_ptr` (`(m, 2*mlp_hidden)`, `[gate;up]` columns), then
      `silu_glu_merged_fp16` computes `silu(gate)*up` into `gated_ptr`.
    - `CutlassFp16SwiGluMlp` (opportunities.md OPT-013): already fuses
      SiLU into its own gate GEMM's epilogue and the gate-multiply into
      its own up GEMM's epilogue -- writes the final gated result
      DIRECTLY into `gated_ptr`, `merged_ptr` unused (no separate
      merged buffer or elementwise kernel needed).
    - `Nvfp4SwiGluMlp` (opportunities.md op-fusion audit finding 2): two
      separate NVFP4 GEMMs each producing an FP4-PACKED intermediate,
      combined by a TRUE-SiLU FP4 kernel straight into `gated_ptr` --
      `merged_ptr` unused here too, no fp16-width merged buffer ever
      materializes.
    """
    mlp_weight = key(gate_slot)
    if isinstance(mlp_weight, (CutlassFp16SwiGluMlp, Nvfp4SwiGluMlp)):
        mlp_weight(modded_ptr, gated_ptr, m, stream)
    else:
        mlp_weight(modded_ptr, merged_ptr, m, stream)
        fvk.silu_glu_merged_fp16(merged_ptr, gated_ptr, m, mlp_hidden, stream)


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


def fp16_adaln_operands(shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, rows: int,
                        dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public interface of `_fuse_mod_group` for other modules: the fp16
    `(dim,)` shift and scale and the `(rows, dim)` materialized gate that the
    unfused AdaLN and gated-residual kernels read, from `(1, 1, dim)` FP32
    modulation chunks. The caller keeps the returned tensors alive while
    any kernel reads their pointers."""
    return _fuse_mod_group(shift, scale, gate, rows, dim)


def fp16_adaln_shift_scale(shift: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Public interface of `_fuse_mod_pair`: the fp16 `(dim,)` shift and
    scale of a gate-less AdaLN pair. Same lifetime contract as
    `fp16_adaln_operands`."""
    return _fuse_mod_pair(shift, scale)


def _add_inplace(dst_ptr: int, src_ptr: int, seq: int, dim: int) -> None:
    """Plain `dst += src`, in place -- used to sum the single-stream
    block's attn-out and mlp-down projections before their ONE shared
    gated residual (see module docstring)."""
    dst = _wrap_fp16(dst_ptr, seq, dim)
    src = _wrap_fp16(src_ptr, seq, dim)
    dst.add_(src)


def _merge_linear2(dims: dict) -> bool:
    """`dims["merge_linear2"]` (roadmap item 4): single-stream blocks run
    the real `linear2` as ONE GEMM over `[attn_out | mlp_act]`. Only the
    merged-`linear1` path writes its SiLU-GLU output into that GEMM's
    input buffer, so the flag requires `dims["merge_qkv_mlp"]`."""
    merge = bool(dims.get("merge_linear2"))
    if merge and not dims.get("merge_qkv_mlp"):
        raise ValueError("dims['merge_linear2'] requires dims['merge_qkv_mlp']")
    return merge


@dataclass(frozen=True)
class AdaLNTarget:
    """The AdaLN step a layer's LAST gated residual update also performs
    when `dims["fuse_res_norm"]` is set (roadmap item 3): the next
    sub-block's `LN_no_affine(residual) * (1 + scale) + shift`, written
    as FP16 to `out_ptr`. `shift`/`scale` are `(1, 1, dim)` FP32 chunks
    straight from `compute_*_modulation`; `out_ptr` is row-aligned with
    the residual rows being updated.

    `lin` (OPT-032 candidate 3, roadmap item 3's own follow-on): the
    GEMM object that consumes this AdaLN's output, attached by
    `_awq_target` (which already receives it). When `lin` is an
    `Nvfp4Linear` and the caller opts in via `dims["fuse_res_norm_fp4"]`,
    `_fused_gate_res` writes NVFP4+SFA directly into `lin`'s own
    activation scratch instead of `out_ptr` (which then holds no valid
    data), and the consumer must read it via `lin.gemm_prequantized(...)`
    instead of `lin(out_ptr, ...)`. `None` when this AdaLN's own layer
    doesn't fold AWQ or opt into candidate 3 (`out_ptr` is the sole
    contract in that case, as before this field existed)."""
    shift: torch.Tensor
    scale: torch.Tensor
    out_ptr: int
    lin: object = None


def _fuse_res_norm(dims: dict, input_normed: bool, *targets: AdaLNTarget | None) -> bool:
    """`dims["fuse_res_norm"]` (roadmap item 3): every gated residual
    update runs `gate_res_ada_layer_norm_*`, which also emits the AdaLN
    that follows it. `input_normed`/`next_*` (the cross-layer chain built
    by `imagewam_prefill`/`imagewam_denoise_step`) require the flag."""
    fuse = bool(dims.get("fuse_res_norm"))
    if not fuse and (input_normed or any(t is not None for t in targets)):
        raise ValueError("input_normed / next AdaLN targets require dims['fuse_res_norm']")
    return fuse


def _mod_vec_ptr(t: torch.Tensor, dim: int) -> int:
    """Device pointer of one `(1, 1, dim)` FP32 modulation chunk, which
    the fused kernel reads as a contiguous `(dim,)` vector."""
    if t.dtype != torch.float32 or t.numel() != dim or t.stride(-1) != 1:
        raise ValueError(f"modulation chunk must be FP32 with {dim} contiguous elements, got "
                         f"dtype={t.dtype} shape={tuple(t.shape)} stride={t.stride()}")
    return t.data_ptr()


def _fused_gate_res(fvk, proj_ptr: int, gate: torch.Tensor, residual_ptr: int, rows: int, dim: int,
                     target: AdaLNTarget | None, stream: int, *, bf16_residual: bool, eps: float,
                     fp4_direct: bool = False) -> None:
    """`residual += gate * proj`, then (if `target`) the next AdaLN into
    `target.out_ptr`, in ONE kernel (roadmap item 3). Bit-identical to
    `gate_res_*` + `ada_layer_norm_*` on the FP16 modulation copies
    `_fuse_mod_group` builds; reads the FP32 modulation directly, so no
    per-layer cast/broadcast kernels are needed.

    `fp4_direct` (OPT-032 candidate 3, opt-in, default off): when True
    AND `target.lin` is an `Nvfp4Linear`, writes NVFP4-packed bytes +
    CUTLASS SFA scale factors directly into `target.lin`'s own
    activation scratch via `csrc/kernels/fused_norm_fp4/`, instead of
    the plain FP16 AdaLN this function otherwise writes to
    `target.out_ptr` -- no intermediate FP16 buffer is ever
    materialized for that consumer. The caller must be `dims`-consistent
    (pass the same `fp4_direct` value used to skip the corresponding
    `Nvfp4Linear.__call__`'s own quantize step at the consuming layer,
    replacing it with `.gemm_prequantized`); `target.out_ptr` holds no
    valid data in this branch. Bit-exact against the unfused
    `gate_res_*` + `quantize_fp4_dynamic_sfa_fp16` pair, confirmed on
    Thor (THOR_CHECKLIST.md X6)."""
    if fp4_direct and target is not None and isinstance(target.lin, Nvfp4Linear):
        lin = target.lin
        lin._ensure_scratch(rows)
        fvk_fp4 = lin._fvk_fp4
        fp4_kernel = (fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res if bf16_residual
                      else fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_fp16res)
        inv_s_ptr = lin.awq_inv_s.data_ptr() if lin.awq_inv_s is not None else 0
        fp4_kernel(residual_ptr, proj_ptr, _mod_vec_ptr(gate, dim), _mod_vec_ptr(target.scale, dim),
                   _mod_vec_ptr(target.shift, dim), inv_s_ptr,
                   lin.scratch.packed.data_ptr(), lin.scratch.sfa.data_ptr(),
                   rows, dim, eps, stream)
        return
    kernel = fvk.gate_res_ada_layer_norm_bf16res if bf16_residual else fvk.gate_res_ada_layer_norm_fp16
    if target is None:
        kernel(proj_ptr, _mod_vec_ptr(gate, dim), residual_ptr, 0, 0, 0, rows, dim, eps, stream)
    else:
        kernel(proj_ptr, _mod_vec_ptr(gate, dim), residual_ptr, _mod_vec_ptr(target.scale, dim),
               _mod_vec_ptr(target.shift, dim), target.out_ptr, rows, dim, eps, stream)


def _awq_input_scaled(lin: object) -> bool:
    """Whether the GEMM `lin` carries an AWQ input scale `s`, so its input
    must be `x / s` (fold A, `flash_rt/models/imagewam/awq.py`)."""
    return isinstance(lin, AwqScaledLinear) and lin.awq_inv_s is not None


def _awq_folded(lin, shift, scale, shift_t, scale_t):
    """AWQ fold A for a standalone AdaLN (`ada_layer_norm_*` with FP16
    `scale`/`shift`): when the GEMM that consumes its output carries an
    AWQ input scale `s`, the pair becomes `(shift / s, (1 + scale) / s - 1)`.
    Otherwise returns `(shift_t, scale_t)` unchanged. `shift`/`scale` are
    the FP32 modulation tensors the folded pair is computed from."""
    if _awq_input_scaled(lin):
        return lin.folded_modulation(shift, scale)
    return shift_t, scale_t


def _awq_target(lin, target: AdaLNTarget | None) -> AdaLNTarget | None:
    """AWQ fold A for an AdaLN emitted by the fused gated residual
    (`dims["fuse_res_norm"]`): the target's FP32 modulation pair replaced
    by the folded one when `lin` (the GEMM consuming `target.out_ptr`)
    carries an AWQ input scale. The fused kernel rounds it to FP16 like
    the standalone path, so both give the same AdaLN output.

    Always attaches `lin` to the returned target (`AdaLNTarget.lin`),
    AWQ-scaled or not -- OPT-032 candidate 3 needs to know the consuming
    GEMM object regardless of AWQ, to detect an `Nvfp4Linear` consumer."""
    if target is None:
        return None
    if not _awq_input_scaled(lin):
        return AdaLNTarget(target.shift, target.scale, target.out_ptr, lin=lin)
    shift_f, scale_f = lin.folded_modulation_fp32(target.shift, target.scale)
    return AdaLNTarget(shift_f, scale_f, target.out_ptr, lin=lin)


def _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn,
                          mod_txt, mod_img, rope_table, *, input_normed: bool = False,
                          next_txt: AdaLNTarget | None = None, next_img: AdaLNTarget | None = None):
    """One real FLUX.2 double-stream block: separate img/txt fused-QKV
    GEMM + MLP, joint attn. Reads/writes `bufs["backbone_hidden"]` rows [0,x0)
    (text) and [x0,a0) (image) in place.

    `dims["fuse_res_norm"]` (roadmap item 3): each gated residual update
    also emits the AdaLN that follows it -- AdaLN2 of this block, and
    `next_txt`/`next_img` (the next block's AdaLN1, into the modded rows
    of each side) after the MLP. `input_normed`: the modded rows already
    hold this block's AdaLN1 output (written by the previous block).
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
    # OPT-032 candidate 1, same flag as _single_stream_layer above. Here
    # RMSNorm is per-stream (txt then img, separate source qkv buffers,
    # separate query_norm/key_norm weights) so the fused kernel is
    # called TWICE, once per stream, each with its own row-offset
    # rope_table pointer into the shared [txt|img] table -- proven
    # equivalent to the reference's separate-norm-then-joint-RoPE
    # sequence by tests/test_fused_qkv_norm_rope_kernel.py::
    # test_double_stream_split_call_matches_joint_rope (RoPE is
    # row-independent/elementwise, so splitting it by row range does
    # not change the result).
    fused_qkv = bool(dims.get("fuse_qkv_norm_rope"))

    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = mod_txt
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = mod_img
    fuse = _fuse_res_norm(dims, input_normed, next_txt, next_img)
    if fuse:
        # Roadmap item 3: the fused kernels read gate/scale/shift from the
        # FP32 modulation directly; FP16 copies are only needed for a
        # standalone AdaLN1 at the start of the chain.
        if not input_normed:
            txt_shift1_t, txt_scale1_t = _fuse_mod_pair(txt_shift1, txt_scale1)
            img_shift1_t, img_scale1_t = _fuse_mod_pair(img_shift1, img_scale1)
            txt_shift1_t, txt_scale1_t = _awq_folded(key("txt_qkv.weight"), txt_shift1, txt_scale1,
                                                     txt_shift1_t, txt_scale1_t)
            img_shift1_t, img_scale1_t = _awq_folded(key("img_qkv.weight"), img_shift1, img_scale1,
                                                     img_shift1_t, img_scale1_t)
    else:
        # OPT-004 step 3: fuse LN+modulate and gated-residual into existing
        # FlashRT kernels -- see _fuse_mod_group's own docstring for why
        # these tensors must stay referenced as locals through this whole
        # function (dangling-pointer safety), and why this Python-side cost
        # is graph-capture-only, never per-replay.
        txt_shift1_t, txt_scale1_t, txt_gate1_t = _fuse_mod_group(txt_shift1, txt_scale1, txt_gate1, x0, hidden)
        txt_shift2_t, txt_scale2_t, txt_gate2_t = _fuse_mod_group(txt_shift2, txt_scale2, txt_gate2, x0, hidden)
        img_shift1_t, img_scale1_t, img_gate1_t = _fuse_mod_group(img_shift1, img_scale1, img_gate1, img_len, hidden)
        img_shift2_t, img_scale2_t, img_gate2_t = _fuse_mod_group(img_shift2, img_scale2, img_gate2, img_len, hidden)
        txt_shift1_t, txt_scale1_t = _awq_folded(key("txt_qkv.weight"), txt_shift1, txt_scale1,
                                                 txt_shift1_t, txt_scale1_t)
        txt_shift2_t, txt_scale2_t = _awq_folded(key("txt_mlp0.weight"), txt_shift2, txt_scale2,
                                                 txt_shift2_t, txt_scale2_t)
        img_shift1_t, img_scale1_t = _awq_folded(key("img_qkv.weight"), img_shift1, img_scale1,
                                                 img_shift1_t, img_scale1_t)
        img_shift2_t, img_scale2_t = _awq_folded(key("img_mlp0.weight"), img_shift2, img_scale2,
                                                 img_shift2_t, img_scale2_t)

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
    if not input_normed:
        fvk.ada_layer_norm_bf16in_fp16out(txt_x, txt_scale1_t.data_ptr(), txt_shift1_t.data_ptr(), modded, x0, hidden, eps, stream)
    txt_qkv_merged = bufs["txt_qkv_merged"]  # (x0, 3*hidden)
    key("txt_qkv.weight")(modded, txt_qkv_merged, x0, stream)
    if fused_qkv:
        fvk.qkv_split_norm_rope_fp16(
            txt_qkv_merged, key("txt_query_norm"), key("txt_key_norm"), rope_table,
            Q_O, K_cache, V_cache, x0, NH, HD, hidden, 3 * hidden,
            0, hidden, 2 * hidden, hidden, eps, stream)
    else:
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

    if not input_normed:
        fvk.ada_layer_norm_bf16in_fp16out(img_x_ptr, img_scale1_t.data_ptr(), img_shift1_t.data_ptr(),
                                 img_modded_ptr, img_len, hidden, eps, stream)
    img_qkv_merged = bufs["img_qkv_merged"]  # (img_len, 3*hidden)
    key("img_qkv.weight")(img_modded_ptr, img_qkv_merged, img_len, stream)
    if fused_qkv:
        img_rope_ptr = _ptr_offset(rope_table, x0, HD)  # row-offset into the shared [txt|img] table
        fvk.qkv_split_norm_rope_fp16(
            img_qkv_merged, key("img_query_norm"), key("img_key_norm"), img_rope_ptr,
            img_Q_ptr, img_K_ptr, img_V_ptr, img_len, NH, HD, hidden, 3 * hidden,
            0, hidden, 2 * hidden, hidden, eps, stream)
    else:
        _copy_slice(img_Q_ptr, img_qkv_merged, img_len, hidden, src_row_stride=3 * hidden)
        _copy_slice(img_K_ptr, _col_ptr(img_qkv_merged, hidden), img_len, hidden, src_row_stride=3 * hidden)
        _copy_slice(img_V_ptr, _col_ptr(img_qkv_merged, 2 * hidden), img_len, hidden, src_row_stride=3 * hidden)
        fvk.rms_norm_fp16(img_Q_ptr, key("img_query_norm"), img_Q_ptr, img_len * NH, HD, eps, stream)
        fvk.rms_norm_fp16(img_K_ptr, key("img_key_norm"), img_K_ptr, img_len * NH, HD, eps, stream)

    if not fused_qkv:
        # RoPE over the FULL combined [txt|img] sequence, once each for Q
        # and K (matches real_double_stream_block_forward_fp16). When
        # fused_qkv is on, RoPE was already applied per-stream above.
        fvk.rope_apply_fp16_perhead(Q_O, rope_table, a0, NH, HD, stream)
        fvk.rope_apply_fp16_perhead(K_cache, rope_table, a0, NH, HD, stream)

    # --- joint self-attention over the whole [text | image] sequence,
    # real per-head, no mask (opportunities.md OPT-002's correction) ---
    attn.run("backbone", layer_idx, q_seq=a0, stream=stream)

    # --- separate output projections, GATED residual ---
    proj = bufs["proj_scratch"]
    key("txt_proj.weight")(Q_O, proj, x0, stream)
    if fuse:
        _fused_gate_res(fvk, proj, txt_gate1, txt_x, x0, hidden,
                        _awq_target(key("txt_mlp0.weight"), AdaLNTarget(txt_shift2, txt_scale2, modded)),
                        stream, bf16_residual=True, eps=eps)
    else:
        fvk.gate_res_bf16res(proj, txt_gate1_t.data_ptr(), txt_x, x0 * hidden, stream)

    img_proj_ptr = _ptr_offset(proj, x0, hidden)
    key("img_proj.weight")(img_Q_ptr, img_proj_ptr, img_len, stream)
    if fuse:
        _fused_gate_res(fvk, img_proj_ptr, img_gate1, img_x_ptr, img_len, hidden,
                        _awq_target(key("img_mlp0.weight"), AdaLNTarget(img_shift2, img_scale2, img_modded_ptr)),
                        stream, bf16_residual=True, eps=eps)
    else:
        fvk.gate_res_bf16res(img_proj_ptr, img_gate1_t.data_ptr(), img_x_ptr, img_len * hidden, stream)

    # --- separate real SiLU-GLU MLPs, GATED residual ---
    if not fuse:
        fvk.ada_layer_norm_bf16in_fp16out(txt_x, txt_scale2_t.data_ptr(), txt_shift2_t.data_ptr(), modded, x0, hidden, eps, stream)
    txt_mlp_merged, txt_mlp_gated = bufs["txt_mlp_merged"], bufs["txt_mlp_gated"]
    _mlp_gate_up(fvk, key, "txt_mlp0.weight", modded, txt_mlp_merged, txt_mlp_gated, x0, mlp_hidden, stream)
    key("txt_mlp2.weight")(txt_mlp_gated, proj, x0, stream)
    if fuse:
        _fused_gate_res(fvk, proj, txt_gate2, txt_x, x0, hidden, next_txt, stream, bf16_residual=True, eps=eps)
    else:
        fvk.gate_res_bf16res(proj, txt_gate2_t.data_ptr(), txt_x, x0 * hidden, stream)

    if not fuse:
        fvk.ada_layer_norm_bf16in_fp16out(img_x_ptr, img_scale2_t.data_ptr(), img_shift2_t.data_ptr(),
                                 img_modded_ptr, img_len, hidden, eps, stream)
    img_mlp_merged, img_mlp_gated = bufs["img_mlp_merged"], bufs["img_mlp_gated"]
    _mlp_gate_up(fvk, key, "img_mlp0.weight", img_modded_ptr, img_mlp_merged, img_mlp_gated,
                 img_len, mlp_hidden, stream)
    key("img_mlp2.weight")(img_mlp_gated, img_proj_ptr, img_len, stream)
    if fuse:
        _fused_gate_res(fvk, img_proj_ptr, img_gate2, img_x_ptr, img_len, hidden, next_img, stream,
                        bf16_residual=True, eps=eps)
    else:
        fvk.gate_res_bf16res(img_proj_ptr, img_gate2_t.data_ptr(), img_x_ptr, img_len * hidden, stream)


def _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn, mod_single, rope_table, *,
                          input_normed: bool = False, next_norm: AdaLNTarget | None = None):
    """One real FLUX.2 single-stream block: merged img+txt, fused
    `qkv` GEMM + separate `mlp_in` GEMM (both real fused `linear1`
    slices, see `real_single_stream_block.py`'s own docstring).
    Operates on the whole `bufs["backbone_hidden"]` (a0, hidden)
    buffer. ``weight_layer_idx`` (0..19) indexes this stream's own
    declared weights; ``site_layer_idx`` continues after the
    double-stream layers, indexing the "backbone" attention site's
    shared 25-layer KV cache.

    `dims["fuse_res_norm"]` (roadmap item 3): `input_normed` -- the
    modded buffer already holds this block's AdaLN output; `next_norm`
    -- the gated residual update also emits the next block's AdaLN.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    NH = dims["NH"]
    mlp_hidden = dims["mlp_hidden"]
    a0 = dims["a0"]
    eps = 1e-6
    key = lambda slot: weights[("backbone", "single", weight_layer_idx, slot)]
    shift, scale, gate = mod_single
    fuse = _fuse_res_norm(dims, input_normed, next_norm)
    if fuse:
        if not input_normed:
            shift_t, scale_t = _fuse_mod_pair(shift, scale)
    else:
        shift_t, scale_t, gate_t = _fuse_mod_group(shift, scale, gate, a0, hidden)
    if dims.get("merge_qkv_mlp") and not input_normed:
        shift_t, scale_t = _awq_folded(key("linear1.weight"), shift, scale, shift_t, scale_t)

    combined = bufs["backbone_hidden"]
    modded = bufs["modded_scratch"]

    ptrs = attn.get_slot_ptrs("backbone", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

    merge_linear2 = _merge_linear2(dims)
    linear2_width = hidden + mlp_hidden
    # OPT-032 candidate 1: one kernel for the Q/K/V split + QK-RMSNorm + RoPE
    # sequence below, in place of 3 strided copies + 2 rms_norm_fp16 + 2
    # rope_apply_fp16_perhead. Opt-in, not yet a default (Thor A/B pending);
    # bit-exact against the unfused sequence on Ada and on Thor
    # (csrc/kernels/fused_qkv_norm_rope/, opportunities.md OPT-032).
    fused_qkv = bool(dims.get("fuse_qkv_norm_rope"))
    # OPT-032 candidate 3, opt-in, Round 1 scope: only this layer's own
    # `merge_qkv_mlp=True` -> `linear1.weight` consumer (the real
    # deployment default and the single biggest/widest GEMM in this
    # function). See `_fused_gate_res`'s own docstring for the contract;
    # other consumers (double-stream targets, the ActionDiT chain, the
    # head target) are not wired yet -- plan.md Phase 4.
    fp4_direct = bool(dims.get("fuse_res_norm_fp4"))

    if not input_normed:
        fvk.ada_layer_norm_bf16in_fp16out(combined, scale_t.data_ptr(), shift_t.data_ptr(), modded, a0, hidden, eps, stream)

    if dims.get("merge_qkv_mlp"):
        # op-fusion audit finding 1: the real fused `linear1` (qkv+
        # mlp-gate/up in ONE GEMM) run once, then Q/K/V and the mlp
        # gate/up columns read directly out of its output via strided
        # views -- no separate `mlp_in` GEMM.
        linear1_width = 3 * hidden + 2 * mlp_hidden
        linear1_out = bufs["single_linear1_merged"]  # (a0, linear1_width)
        linear1 = key("linear1.weight")
        if fp4_direct and input_normed and isinstance(linear1, Nvfp4Linear):
            # OPT-032 candidate 3: the previous layer's `_fused_gate_res`
            # already wrote NVFP4+SFA straight into `linear1`'s own
            # scratch (see that function's `fp4_direct` branch) -- `modded`
            # holds no valid data for this call, read nothing from it.
            linear1.gemm_prequantized(linear1_out, a0, stream)
        else:
            linear1(modded, linear1_out, a0, stream)
        if fused_qkv:
            fvk.qkv_split_norm_rope_fp16(
                linear1_out, key("query_norm"), key("key_norm"), rope_table,
                Q_O, K_cache, V_cache, a0, NH, HD, hidden, linear1_width,
                0, hidden, 2 * hidden, hidden, eps, stream)
        else:
            _copy_slice(Q_O, linear1_out, a0, hidden, src_row_stride=linear1_width)
            _copy_slice(K_cache, _col_ptr(linear1_out, hidden), a0, hidden, src_row_stride=linear1_width)
            _copy_slice(V_cache, _col_ptr(linear1_out, 2 * hidden), a0, hidden, src_row_stride=linear1_width)
        if merge_linear2:
            # Roadmap item 4: the SiLU-GLU output lands in the MLP
            # columns of the merged `linear2` input `[attn_out | mlp_act]`.
            linear2_in = bufs["single_linear2_in"]  # (a0, linear2_width)
            fvk.silu_glu_merged_fp16(_col_ptr(linear1_out, 3 * hidden), _col_ptr(linear2_in, hidden),
                                      a0, mlp_hidden, stream, linear1_width, linear2_width)
        else:
            mlp_gated = bufs["single_mlp_gated"]
            fvk.silu_glu_merged_fp16(_col_ptr(linear1_out, 3 * hidden), mlp_gated, a0, mlp_hidden, stream,
                                      linear1_width)
    else:
        qkv_merged = bufs["single_qkv_merged"]  # (a0, 3*hidden)
        key("qkv.weight")(modded, qkv_merged, a0, stream)
        if fused_qkv:
            fvk.qkv_split_norm_rope_fp16(
                qkv_merged, key("query_norm"), key("key_norm"), rope_table,
                Q_O, K_cache, V_cache, a0, NH, HD, hidden, 3 * hidden,
                0, hidden, 2 * hidden, hidden, eps, stream)
        else:
            _copy_slice(Q_O, qkv_merged, a0, hidden, src_row_stride=3 * hidden)
            _copy_slice(K_cache, _col_ptr(qkv_merged, hidden), a0, hidden, src_row_stride=3 * hidden)
            _copy_slice(V_cache, _col_ptr(qkv_merged, 2 * hidden), a0, hidden, src_row_stride=3 * hidden)
        mlp_merged, mlp_gated = bufs["single_mlp_merged"], bufs["single_mlp_gated"]
        _mlp_gate_up(fvk, key, "mlp_in.weight", modded, mlp_merged, mlp_gated, a0, mlp_hidden, stream)

    if not fused_qkv:
        fvk.rms_norm_fp16(Q_O, key("query_norm"), Q_O, a0 * NH, HD, eps, stream)
        fvk.rms_norm_fp16(K_cache, key("key_norm"), K_cache, a0 * NH, HD, eps, stream)
        fvk.rope_apply_fp16_perhead(Q_O, rope_table, a0, NH, HD, stream)
        fvk.rope_apply_fp16_perhead(K_cache, rope_table, a0, NH, HD, stream)

    attn.run("backbone", site_layer_idx, q_seq=a0, stream=stream)

    from_attn = bufs["proj_scratch"]
    if merge_linear2:
        # Roadmap item 4: attention output joins the MLP activation in
        # the `linear2` input, then ONE GEMM with K = hidden + mlp_hidden.
        _copy_slice(linear2_in, Q_O, a0, hidden, dst_row_stride=linear2_width)
        key("linear2.weight")(linear2_in, from_attn, a0, stream)
    else:
        from_mlp = bufs["proj_scratch2"]
        key("attn_out_proj.weight")(Q_O, from_attn, a0, stream)
        key("mlp_down.weight")(mlp_gated, from_mlp, a0, stream)
        _add_inplace(from_attn, from_mlp, a0, hidden)
    if fuse:
        _fused_gate_res(fvk, from_attn, gate, combined, a0, hidden, next_norm, stream, bf16_residual=True,
                        eps=eps, fp4_direct=fp4_direct)
    else:
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
    num_single = dims["num_layers_single"]
    # Roadmap item 3 (`dims["fuse_res_norm"]`): each layer's last gated
    # residual update also writes the NEXT layer's AdaLN output into
    # `modded_scratch` (txt rows and img rows separately; LayerNorm is
    # per row, so the last double layer can emit the single blocks'
    # AdaLN for both). The last single layer has no following AdaLN.
    fuse = bool(dims.get("fuse_res_norm"))
    modded = bufs["modded_scratch"]
    img_modded = _ptr_offset(modded, x0, hidden)
    (txt_shift1, txt_scale1, _), _ = mod_txt
    (img_shift1, img_scale1, _), _ = mod_img
    single_shift, single_scale, _ = mod_single
    # AWQ fold A (awq.py): each emitted AdaLN is folded for the GEMM that
    # consumes it -- the next double layer's qkv, or single layer 0's /
    # the next single layer's linear1.
    single_linear1 = lambda i: weights.get(("backbone", "single", i, "linear1.weight"))
    for layer_idx in range(num_double):
        next_txt = next_img = None
        if fuse and layer_idx + 1 < num_double:
            nxt = lambda slot: weights[("backbone", "double", layer_idx + 1, slot)]
            next_txt = _awq_target(nxt("txt_qkv.weight"), AdaLNTarget(txt_shift1, txt_scale1, modded))
            next_img = _awq_target(nxt("img_qkv.weight"), AdaLNTarget(img_shift1, img_scale1, img_modded))
        elif fuse and num_single > 0:
            next_txt = _awq_target(single_linear1(0), AdaLNTarget(single_shift, single_scale, modded))
            next_img = _awq_target(single_linear1(0), AdaLNTarget(single_shift, single_scale, img_modded))
        _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn,
                              mod_txt, mod_img, rope_table, input_normed=fuse and layer_idx > 0,
                              next_txt=next_txt, next_img=next_img)
    # The "backbone" attention site's per-layer KV cache is ONE
    # contiguous 25-layer range (num_layers_double + num_layers_single);
    # single-stream layers continue that same indexing rather than
    # restarting at 0, which would otherwise alias double-stream layer
    # 0..4's own K/V cache slots.
    for i in range(num_single):
        next_norm = (_awq_target(single_linear1(i + 1), AdaLNTarget(single_shift, single_scale, modded))
                     if fuse and i + 1 < num_single else None)
        _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn,
                              mod_single, rope_table, input_normed=fuse and (num_double > 0 or i > 0),
                              next_norm=next_norm)


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
                          mod, action_rope_table, *, input_normed: bool = False,
                          next_norm: AdaLNTarget | None = None):
    """ActionDiT double block (img-only). `dims["fuse_res_norm"]`: same
    `input_normed`/`next_norm` contract as `_double_stream_layer`, FP16
    residual."""
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
    fuse = _fuse_res_norm(dims, input_normed, next_norm)
    if fuse:
        if not input_normed:
            shift1_t, scale1_t = _fuse_mod_pair(shift1, scale1)
            shift1_t, scale1_t = _awq_folded(key("qkv.weight"), shift1, scale1, shift1_t, scale1_t)
    else:
        shift1_t, scale1_t, gate1_t = _fuse_mod_group(shift1, scale1, gate1, num_action, action_hidden_dim)
        shift2_t, scale2_t, gate2_t = _fuse_mod_group(shift2, scale2, gate2, num_action, action_hidden_dim)
        shift1_t, scale1_t = _awq_folded(key("qkv.weight"), shift1, scale1, shift1_t, scale1_t)
        shift2_t, scale2_t = _awq_folded(key("mlp0.weight"), shift2, scale2, shift2_t, scale2_t)

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
    # OPT-032 candidate 1, same as _single_stream_layer/_action_single_layer.
    fused_qkv = bool(dims.get("fuse_qkv_norm_rope"))

    if not input_normed:
        fvk.ada_layer_norm_fp16(action_x, scale1_t.data_ptr(), shift1_t.data_ptr(),
                                 modded, num_action, action_hidden_dim, eps, stream)
    qkv_merged = bufs["action_qkv_merged"]  # (num_action, 3*action_attn_width)
    key("qkv.weight")(modded, qkv_merged, num_action, stream)
    if fused_qkv:
        fvk.qkv_split_norm_rope_fp16(
            qkv_merged, key("query_norm"), key("key_norm"), action_rope_table,
            action_Q_ptr, action_K_ptr, action_V_ptr, num_action, NH, HD, action_attn_width,
            3 * action_attn_width, 0, action_attn_width, 2 * action_attn_width, action_attn_width, eps, stream)
    else:
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
    if fuse:
        _fused_gate_res(fvk, proj, gate1, action_x, num_action, action_hidden_dim,
                        _awq_target(key("mlp0.weight"), AdaLNTarget(shift2, scale2, modded)), stream,
                        bf16_residual=False, eps=eps)
    else:
        fvk.gate_res_fp16(proj, gate1_t.data_ptr(), action_x, num_action * action_hidden_dim, stream)
        fvk.ada_layer_norm_fp16(action_x, scale2_t.data_ptr(), shift2_t.data_ptr(),
                                 modded, num_action, action_hidden_dim, eps, stream)
    mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
    _mlp_gate_up(fvk, key, "mlp0.weight", modded, mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)
    key("mlp2.weight")(mlp_gated, proj, num_action, stream)
    if fuse:
        _fused_gate_res(fvk, proj, gate2, action_x, num_action, action_hidden_dim, next_norm, stream,
                        bf16_residual=False, eps=eps)
    else:
        fvk.gate_res_fp16(proj, gate2_t.data_ptr(), action_x, num_action * action_hidden_dim, stream)


def _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn, mod, action_rope_table, *,
                          input_normed: bool = False, next_norm: AdaLNTarget | None = None):
    """ActionDiT single block. `dims["fuse_res_norm"]`: same
    `input_normed`/`next_norm` contract as `_single_stream_layer`, FP16
    residual (the last block's `next_norm` is the head's AdaLN)."""
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
    fuse = _fuse_res_norm(dims, input_normed, next_norm)
    if fuse:
        if not input_normed:
            shift_t, scale_t = _fuse_mod_pair(shift, scale)
    else:
        shift_t, scale_t, gate_t = _fuse_mod_group(shift, scale, gate, num_action, action_hidden_dim)
    if dims.get("merge_qkv_mlp") and not input_normed:
        shift_t, scale_t = _awq_folded(key("linear1.weight"), shift, scale, shift_t, scale_t)

    action_x = bufs["action_hidden"]
    modded = bufs["action_modded"]

    ptrs = attn.get_slot_ptrs("mot", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
    action_Q_ptr = _ptr_offset(Q_O, a0, action_attn_width)
    action_K_ptr = _ptr_offset(K_cache, a0, action_attn_width)
    action_V_ptr = _ptr_offset(V_cache, a0, action_attn_width)

    merge_linear2 = _merge_linear2(dims)
    linear2_width = action_attn_width + action_mlp_hidden
    # OPT-032 candidate 1, same as _single_stream_layer above.
    fused_qkv = bool(dims.get("fuse_qkv_norm_rope"))

    if not input_normed:
        fvk.ada_layer_norm_fp16(action_x, scale_t.data_ptr(), shift_t.data_ptr(),
                                 modded, num_action, action_hidden_dim, eps, stream)
    if dims.get("merge_qkv_mlp"):
        # op-fusion audit finding 1: same real fused `linear1` merge as
        # `_single_stream_layer` above, for ActionDiT's own single-
        # stream blocks.
        linear1_width = 3 * action_attn_width + 2 * action_mlp_hidden
        linear1_out = bufs["action_linear1_merged"]  # (num_action, linear1_width)
        key("linear1.weight")(modded, linear1_out, num_action, stream)
        if fused_qkv:
            fvk.qkv_split_norm_rope_fp16(
                linear1_out, key("query_norm"), key("key_norm"), action_rope_table,
                action_Q_ptr, action_K_ptr, action_V_ptr, num_action, NH, HD, action_attn_width,
                linear1_width, 0, action_attn_width, 2 * action_attn_width, action_attn_width, eps, stream)
        else:
            _copy_slice(action_Q_ptr, linear1_out, num_action, action_attn_width, src_row_stride=linear1_width)
            _copy_slice(action_K_ptr, _col_ptr(linear1_out, action_attn_width), num_action, action_attn_width,
                        src_row_stride=linear1_width)
            _copy_slice(action_V_ptr, _col_ptr(linear1_out, 2 * action_attn_width), num_action, action_attn_width,
                        src_row_stride=linear1_width)
        if merge_linear2:
            # Roadmap item 4: same merged `linear2` input as
            # `_single_stream_layer` above.
            linear2_in = bufs["action_linear2_in"]  # (num_action, linear2_width)
            fvk.silu_glu_merged_fp16(_col_ptr(linear1_out, 3 * action_attn_width),
                                      _col_ptr(linear2_in, action_attn_width),
                                      num_action, action_mlp_hidden, stream, linear1_width, linear2_width)
        else:
            mlp_gated = bufs["action_mlp_gated"]
            fvk.silu_glu_merged_fp16(_col_ptr(linear1_out, 3 * action_attn_width), mlp_gated,
                                      num_action, action_mlp_hidden, stream, linear1_width)
    else:
        qkv_merged = bufs["action_qkv_merged"]  # (num_action, 3*action_attn_width)
        key("qkv.weight")(modded, qkv_merged, num_action, stream)
        if fused_qkv:
            fvk.qkv_split_norm_rope_fp16(
                qkv_merged, key("query_norm"), key("key_norm"), action_rope_table,
                action_Q_ptr, action_K_ptr, action_V_ptr, num_action, NH, HD, action_attn_width,
                3 * action_attn_width, 0, action_attn_width, 2 * action_attn_width, action_attn_width, eps, stream)
        else:
            _copy_slice(action_Q_ptr, qkv_merged, num_action, action_attn_width, src_row_stride=3 * action_attn_width)
            _copy_slice(action_K_ptr, _col_ptr(qkv_merged, action_attn_width), num_action, action_attn_width,
                        src_row_stride=3 * action_attn_width)
            _copy_slice(action_V_ptr, _col_ptr(qkv_merged, 2 * action_attn_width), num_action, action_attn_width,
                        src_row_stride=3 * action_attn_width)
        mlp_merged, mlp_gated = bufs["action_mlp_merged"], bufs["action_mlp_gated"]
        _mlp_gate_up(fvk, key, "mlp_in.weight", modded, mlp_merged, mlp_gated, num_action, action_mlp_hidden, stream)
    if not fused_qkv:
        fvk.rms_norm_fp16(action_Q_ptr, key("query_norm"), action_Q_ptr, num_action * NH, HD, eps, stream)
        fvk.rms_norm_fp16(action_K_ptr, key("key_norm"), action_K_ptr, num_action * NH, HD, eps, stream)
        fvk.rope_apply_fp16_perhead(action_Q_ptr, action_rope_table, num_action, NH, HD, stream)
        fvk.rope_apply_fp16_perhead(action_K_ptr, action_rope_table, num_action, NH, HD, stream)

    attn.run("mot", site_layer_idx, q_seq=num_action, kv_seq=dims["total"], stream=stream, x0=x0, a0=a0)

    from_attn = bufs["action_proj_scratch"]
    if merge_linear2:
        _copy_slice(linear2_in, action_Q_ptr, num_action, action_attn_width, dst_row_stride=linear2_width)
        key("linear2.weight")(linear2_in, from_attn, num_action, stream)
    else:
        from_mlp = bufs["action_proj_scratch2"]
        key("attn_out_proj.weight")(action_Q_ptr, from_attn, num_action, stream)
        key("mlp_down.weight")(mlp_gated, from_mlp, num_action, stream)
        _add_inplace(from_attn, from_mlp, num_action, action_hidden_dim)
    if fuse:
        _fused_gate_res(fvk, from_attn, gate, action_x, num_action, action_hidden_dim, next_norm, stream,
                        bf16_residual=False, eps=eps)
    else:
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
    num_single = dims["action_num_layers_single"]
    # Roadmap item 3 (`dims["fuse_res_norm"]`): same chain as
    # `imagewam_prefill`; the last layer's fused update emits the head's
    # AdaLN into `head_modded`.
    fuse = bool(dims.get("fuse_res_norm"))
    modded = bufs["action_modded"]
    (double_shift1, double_scale1, _), _ = mod_double
    single_shift, single_scale, _ = mod_single
    # AWQ fold A (awq.py): each emitted AdaLN is folded for the GEMM that
    # consumes it (next double qkv, single linear1, head).
    head_target = _awq_target(key("head.linear.weight"), AdaLNTarget(head_mod[0], head_mod[1], bufs["head_modded"]))
    single_linear1 = lambda i: weights.get(("action_dit", "single", i, "linear1.weight"))
    for layer_idx in range(num_double):
        next_norm = None
        if fuse:
            if layer_idx + 1 < num_double:
                next_norm = _awq_target(weights[("action_dit", "double", layer_idx + 1, "qkv.weight")],
                                        AdaLNTarget(double_shift1, double_scale1, modded))
            elif num_single > 0:
                next_norm = _awq_target(single_linear1(0), AdaLNTarget(single_shift, single_scale, modded))
            else:
                next_norm = head_target
        _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, layer_idx, stream, attn,
                              mod_double, action_rope_table, input_normed=fuse and layer_idx > 0,
                              next_norm=next_norm)
    for i in range(num_single):
        next_norm = None
        if fuse:
            next_norm = (_awq_target(single_linear1(i + 1), AdaLNTarget(single_shift, single_scale, modded))
                         if i + 1 < num_single else head_target)
        _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn,
                              mod_single, action_rope_table, input_normed=fuse and (num_double > 0 or i > 0),
                              next_norm=next_norm)

    # --- decode: real head (AdaLN, no gate, + Linear) -> velocity
    # (fp16, action_dim); Euler step in action_dim space, matching
    # imagewam.py's own infer_action_flux2 exactly ---
    if not (fuse and num_double + num_single > 0):
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

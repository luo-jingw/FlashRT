// ================================================================
// FlashRT — Fused QKV column-slice split + RMSNorm(Q,K) + RoPE(Q,K)
// + plain-copy(V), FP16.
//
// Built into flash_rt_kernels (CMakeLists.txt) and bound as
// `qkv_split_norm_rope_fp16` (csrc/bindings.cpp); opt-in in the ImageWAM
// pipeline via dims["fuse_qkv_norm_rope"]; validated by
// tests/test_fused_qkv_norm_rope_kernel.py through that binding.
//
// Reproduces, in ONE kernel launch, the exact 7-call sequence found in
// flash_rt/models/imagewam/pipeline_thor.py's `_single_stream_layer`
// (and the per-stream halves of `_double_stream_layer`, and
// `_action_double_layer`/`_action_single_layer`) with the served default
// `dims["vec_rms_norm"]` on:
//
//   _copy_slice(Q_O,      qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset q_col_offset
//   _copy_slice(K_cache,  qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset k_col_offset
//   _copy_slice(V_cache,  qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset v_col_offset
//   rms_norm_fp16_vec(Q_O,     q_norm_weight, Q_O,     rows*NH, HD, eps, stream)
//   rms_norm_fp16_vec(K_cache, k_norm_weight, K_cache, rows*NH, HD, eps, stream)
//   rope_apply_fp16_perhead(Q_O,     rope_table, rows, NH, HD, stream)
//   rope_apply_fp16_perhead(K_cache, rope_table, rows, NH, HD, stream)
//
// (split Q, K, V, then RMSNorm Q and K [not V], then RoPE Q and K --
// norm before rope, matching pipeline_thor.py).
//
// Numerics are bit-exact against csrc/kernels/vec_fp16_backbone.cu's
// `rms_norm_fp16_vec_warp_kernel` (one warp per row; each lane sums the
// squares of its int4 chunks left to right, then a xor-butterfly warp
// sum; rms = rsqrtf(sum/HD + eps); out = (x*rms)*weight) and
// csrc/kernels/rope.cu's `rope_apply_fp16_perhead_kernel`
// (interleaved-pair RoPE: pair (2d,2d+1) rotated by (cos,sin) =
// rope_table[seq_pos*HD+2d], rope_table[seq_pos*HD+2d+1]). It is NOT
// bit-exact against the 256-thread `rms_norm_fp16` path
// (`dims["vec_rms_norm"]` off): that kernel reduces in a different order.
//
// One warp per (row, head), 8 warps per block. Every shape parameter is
// a runtime argument.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// `qkv`             merged linear1/qkv GEMM output, row-major fp16,
//                    row stride `src_row_stride` elements (e.g.
//                    3*hidden for a plain qkv GEMM, or
//                    3*hidden+2*mlp_hidden for a fused linear1 that
//                    also carries the MLP gate/up columns).
// `q_norm_weight`,
// `k_norm_weight`   (HD,) RMSNorm scale weights.
// `rope_table`      base pointer at THIS call's row 0 (caller offsets
//                    by row*HD*sizeof(__half) for a sub-range, e.g.
//                    the image stream's own starting row within a
//                    double-stream block's combined [txt|img]
//                    sequence) — same interleaved cos/sin-per-pair
//                    layout `rope_apply_fp16_perhead_kernel` reads.
// `Q_out`/`K_out`/
// `V_out`           destination buffers, row stride `dst_row_stride`
//                    elements (Q_O/K_cache/V_cache's own row width:
//                    `hidden` for the backbone, `action_attn_width`
//                    for ActionDiT).
// `rows`            number of rows (tokens) this call covers.
// `NH`, `HD`        heads and per-head dim; `hidden` = NH*HD is the
//                    Q/K/V slice width (kept as an explicit parameter
//                    rather than recomputed, since `dst_row_stride`
//                    is allowed to differ from it in principle).
// `q_col_offset`/
// `k_col_offset`/
// `v_col_offset`    column offset (elements) of each slice within one
//                    `qkv` row (0, hidden, 2*hidden for a plain qkv
//                    GEMM output).
// `eps`             RMSNorm epsilon.
//
// Preconditions (checked; returns -1 and launches nothing if unmet):
// HD, both row strides and all three column offsets are multiples of 8
// elements, and every pointer is 16-byte aligned (int4 loads and
// stores). Real ImageWAM shapes (HD=128, widths 3072/9216) satisfy
// them. Returns 0 on success, or -cudaError_t on a launch error.
int qkv_split_norm_rope_fp16(
    const __half* qkv,
    const __half* q_norm_weight,
    const __half* k_norm_weight,
    const __half* rope_table,
    __half* Q_out,
    __half* K_out,
    __half* V_out,
    int rows, int NH, int HD, int hidden,
    int src_row_stride,
    int q_col_offset, int k_col_offset, int v_col_offset,
    int dst_row_stride,
    float eps,
    cudaStream_t stream);

// ================================================================
// FlashRT — Fused QKV column-slice split + RMSNorm(Q,K) + RoPE(Q,K)
// + plain-copy(V), FP16.
//
// Standalone development kernel (not wired into the production
// pipeline/build — see tests/test_fused_qkv_norm_rope_kernel.py for
// how it is JIT-compiled and validated bit-exact against the existing
// production primitives it fuses).
//
// Reproduces, in ONE kernel launch, the exact 5-call sequence found in
// flash_rt/models/imagewam/pipeline_thor.py's `_single_stream_layer`
// (and the per-stream halves of `_double_stream_layer`, and
// `_action_double_layer`/`_action_single_layer`):
//
//   _copy_slice(Q_O,      qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset q_col_offset
//   _copy_slice(K_cache,  qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset k_col_offset
//   _copy_slice(V_cache,  qkv, rows, hidden, src_row_stride=src_row_stride)  // col offset v_col_offset
//   rms_norm_fp16(Q_O,     q_norm_weight, Q_O,     rows*NH, HD, eps, stream)
//   rms_norm_fp16(K_cache, k_norm_weight, K_cache, rows*NH, HD, eps, stream)
//   rope_apply_fp16_perhead(Q_O,     rope_table, rows, NH, HD, stream)
//   rope_apply_fp16_perhead(K_cache, rope_table, rows, NH, HD, stream)
//
// (split ALL THREE of Q,K,V first, THEN RMSNorm Q and K [not V], THEN
// RoPE Q and K — norm before rope, matching pipeline_thor.py exactly,
// see its module docstring and _single_stream_layer body).
//
// Numerics are matched bit-exact against csrc/kernels/norm.cu's
// `rms_norm_kernel<__half>` (RMSNorm: rms = rsqrtf(mean(x^2) + eps),
// out = (x*rms)*weight, left-to-right) and csrc/kernels/rope.cu's
// `rope_apply_fp16_perhead_kernel` (interleaved-pair RoPE: pair
// (2d,2d+1) rotated by (cos,sin) = rope_table[seq_pos*HD+2d],
// rope_table[seq_pos*HD+2d+1]) — see those files for the ground-truth
// source this kernel was written against.
//
// One CUDA block per (row, head): blockDim.x is FIXED at 256 to match
// rms_norm_fp16's own hardcoded launch config exactly — this is not a
// performance tuning knob, it is required for the block-reduction
// summation order (warp_reduce_sum/block_reduce_sum tree in
// csrc/kernels/common.cuh, reused here unmodified) to stay bit-for-bit
// identical to the reference kernel's own reduction. Every SHAPE
// parameter (rows, NH, HD, hidden, row strides, column offsets) is a
// runtime argument — nothing about the model's per-call-site dims is
// baked in as a compile-time constant.
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
// Precondition: `HD` must be even (Q/K/V are read/written via __half2
// vectorized loads at per-head byte offset head*HD*sizeof(__half),
// which needs 4-byte alignment). Real ImageWAM shapes (backbone and
// ActionDiT) always use HD=128 — this is not a limitation in
// practice, just an unchecked precondition an odd HD would violate.
void qkv_split_norm_rope_fp16(
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

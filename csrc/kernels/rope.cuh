// ================================================================
// FlashRT — RoPE kernel declarations
// Standard RoPE, QKV split, fused QKV split + RoPE
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// RoPE: apply rotary position embeddings to Q and K
void rope_apply(const __nv_bfloat16* rope_weights,
                __nv_bfloat16* Q, __nv_bfloat16* K,
                int seq_len, int num_heads, int head_dim,
                cudaStream_t stream = 0);

// QKV split: split (seq, q+k+v) into separate Q, K, V (BF16)
void qkv_split(const __nv_bfloat16* qkv,
               __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
               int seq, int q_dim, int k_dim, int v_dim,
               cudaStream_t stream = 0);

// QKV split: split (seq, q+k+v) into separate Q, K, V (FP16)
void qkv_split_fp16(const __half* qkv,
                    __half* Q, __half* K, __half* V,
                    int seq, int q_dim, int k_dim, int v_dim,
                    cudaStream_t stream = 0);

// Fused QKV split + RoPE: split and apply RoPE in one kernel
void qkv_split_rope(const __nv_bfloat16* qkv,
                     const __nv_bfloat16* rope_weights,
                     __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                     int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                     cudaStream_t stream = 0);

// Same as qkv_split_rope, but the K/V cache write row is shifted by a RUNTIME
// device offset ``devpos[0]`` (K/V are cache base pointers, row 0). Lets one
// fixed-shape graph append decoder K/V after a variable-length valid prefix.
void qkv_split_rope_devpos(const __nv_bfloat16* qkv,
                           const __nv_bfloat16* rope_weights,
                           __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                           const int* devpos,
                           int seq, int q_dim, int k_dim, int v_dim,
                           int head_dim, cudaStream_t stream = 0);

// Fused QKV split + RoPE + KV cache write (FP16)
// Matches pi05 qkv_split_rope_kvcache_k exactly.
// Q → contiguous (S, Q_dim), K → Kc[kc_offset + s*kc_stride], V → Vc[kc_offset + s*kc_stride]
void qkv_split_rope_kvcache_fp16(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride,
    cudaStream_t stream = 0);

// Same FP16 math and argument contract as qkv_split_rope_kvcache_fp16, but
// K/V write rows are shifted by device-side devpos[0] within the layer slab.
void qkv_split_rope_kvcache_fp16_devpos(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    const int* devpos,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride,
    cudaStream_t stream = 0);

// FP16 RoPE, real per-head (ImageWAM OPT-002 follow-up: unlike
// `rope_apply` above, which assumes K has a single shared head, this
// rotates ONE tensor shaped (seq, NH, HD) in place -- call it once for
// Q and once for K when both are real per-head (see
// csrc/kernels/attention_cublas.cuh for that layout convention).
// `rope_weights` is (seq, HD) with the SAME interleaved-cos/sin-per-
// pair format `rope_apply` uses (rope_weights[pos*HD + 2*d] = cos,
// [pos*HD + 2*d+1] = sin for pair d) -- shared across every head at a
// given position, matching real RoPE (position doesn't depend on head).
void rope_apply_fp16_perhead(
    __half* X, const __half* rope_weights,
    int seq, int NH, int HD,
    cudaStream_t stream = 0);

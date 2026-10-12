// ================================================================
// FlashRT — Fused QKV column-slice split + RMSNorm(Q,K) + RoPE(Q,K)
// + plain-copy(V), FP16. See the .cuh header for the full contract
// and the exact production call sequence this reproduces.
// ================================================================
#include "qkv_split_norm_rope_fp16.cuh"

#include <cstdint>

namespace {

// Same butterfly as csrc/kernels/vec_fp16_backbone.cu's `warp_sum`, so the
// per-head sum of squares is reduced in the same order as
// `rms_norm_fp16_vec_warp_kernel` (the per-head Q/K RMSNorm the served
// default runs, `dims["vec_rms_norm"]`).
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

constexpr int kWarpsPerBlock = 8;

// Sum of squares of one head: lane `lane` accumulates its int4 chunks
// (8 halves each, left to right) exactly as rms_norm_fp16_vec_warp_kernel
// does, then the warp butterfly.
__device__ __forceinline__ float head_sum_sq(const int4* x4, int n4, int lane) {
    float ss = 0.f;
    for (int i = lane; i < n4; i += 32) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = __half2float(hh[j]);
            ss += v * v;
        }
    }
    return warp_sum(ss);
}

// One int4 chunk (4 interleaved pairs): RMSNorm with the same
// (x*rms)*w order as rms_norm_fp16_vec_warp_kernel, rounded to fp16 (the
// reference writes the normed head to memory before RoPE reads it back),
// then rope_apply_fp16_perhead_kernel's pair rotation.
__device__ __forceinline__ int4 norm_rope_chunk(int4 raw, int4 wr, int4 rp, float rms) {
    const __half* hh = reinterpret_cast<const __half*>(&raw);
    const __half* wh = reinterpret_cast<const __half*>(&wr);
    const __half* ph = reinterpret_cast<const __half*>(&rp);
    int4 res;
    __half* rh = reinterpret_cast<__half*>(&res);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float x0 = __half2float(__float2half(__half2float(hh[2 * p]) * rms * __half2float(wh[2 * p])));
        const float x1 = __half2float(__float2half(__half2float(hh[2 * p + 1]) * rms * __half2float(wh[2 * p + 1])));
        const float c = __half2float(ph[2 * p]);
        const float s = __half2float(ph[2 * p + 1]);
        rh[2 * p] = __float2half(x0 * c - x1 * s);
        rh[2 * p + 1] = __float2half(x1 * c + x0 * s);
    }
    return res;
}

// One warp per (row, head).
__global__ void qkv_split_norm_rope_fp16_kernel(
    const __half* __restrict__ qkv,
    const __half* __restrict__ q_norm_weight,
    const __half* __restrict__ k_norm_weight,
    const __half* __restrict__ rope_table,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    int rows, int NH, int HD,
    int src_row_stride,
    int q_col_offset, int k_col_offset, int v_col_offset,
    int dst_row_stride,
    float eps) {
    const int item = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (item >= rows * NH) return;
    const int row = item / NH;
    const int head = item % NH;
    const int lane = threadIdx.x & 31;
    const int n4 = HD >> 3;

    const __half* src = qkv + (long)row * src_row_stride + (long)head * HD;
    const int4* q4 = reinterpret_cast<const int4*>(src + q_col_offset);
    const int4* k4 = reinterpret_cast<const int4*>(src + k_col_offset);
    const int4* v4 = reinterpret_cast<const int4*>(src + v_col_offset);
    const long dst = (long)row * dst_row_stride + (long)head * HD;
    int4* qo4 = reinterpret_cast<int4*>(Q_out + dst);
    int4* ko4 = reinterpret_cast<int4*>(K_out + dst);
    int4* vo4 = reinterpret_cast<int4*>(V_out + dst);
    const int4* qw4 = reinterpret_cast<const int4*>(q_norm_weight);
    const int4* kw4 = reinterpret_cast<const int4*>(k_norm_weight);
    const int4* rope4 = reinterpret_cast<const int4*>(rope_table + (long)row * HD);

    const float q_rms = rsqrtf(head_sum_sq(q4, n4, lane) / HD + eps);
    const float k_rms = rsqrtf(head_sum_sq(k4, n4, lane) / HD + eps);

    for (int i = lane; i < n4; i += 32) {
        const int4 rp = rope4[i];
        qo4[i] = norm_rope_chunk(q4[i], qw4[i], rp, q_rms);
        ko4[i] = norm_rope_chunk(k4[i], kw4[i], rp, k_rms);
        vo4[i] = v4[i];
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

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
    cudaStream_t stream) {
    (void)hidden;  // = NH*HD; kept in the interface, not needed for addressing
    if (HD % 8 != 0 || src_row_stride % 8 != 0 || dst_row_stride % 8 != 0 ||
        q_col_offset % 8 != 0 || k_col_offset % 8 != 0 || v_col_offset % 8 != 0) return -1;
    if (!aligned16(qkv) || !aligned16(q_norm_weight) || !aligned16(k_norm_weight) ||
        !aligned16(rope_table) || !aligned16(Q_out) || !aligned16(K_out) || !aligned16(V_out)) return -1;
    const int items = rows * NH;
    const int blocks = (items + kWarpsPerBlock - 1) / kWarpsPerBlock;
    qkv_split_norm_rope_fp16_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        qkv, q_norm_weight, k_norm_weight, rope_table,
        Q_out, K_out, V_out,
        rows, NH, HD, src_row_stride,
        q_col_offset, k_col_offset, v_col_offset,
        dst_row_stride, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

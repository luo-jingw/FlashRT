// ================================================================
// FlashRT — Fused QKV column-slice split + RMSNorm(Q,K) + RoPE(Q,K)
// + plain-copy(V), FP16. See the .cuh header for the full contract
// and the exact production call sequence this reproduces.
// ================================================================
#include "qkv_split_norm_rope_fp16.cuh"

// Reuses the project's own warp_reduce_sum/block_reduce_sum verbatim
// (unmodified header, included read-only) so the block-reduction
// summation order is bit-for-bit identical to rms_norm_fp16's own
// (csrc/kernels/norm.cu), not merely numerically close.
#include "../common.cuh"

namespace {

// Fixed at 256 to match rms_norm_fp16's own hardcoded launch config
// (csrc/kernels/norm.cu: `<<<seq_len, 256, ...>>>`) — see the .cuh
// header's own comment for why this is a correctness requirement, not
// a tuning knob.
constexpr int kBlockThreads = 256;

__global__ void qkv_split_norm_rope_fp16_kernel(
    const __half* __restrict__ qkv,
    const __half* __restrict__ q_norm_weight,
    const __half* __restrict__ k_norm_weight,
    const __half* __restrict__ rope_table,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    int NH, int HD, int hidden,
    int src_row_stride,
    int q_col_offset, int k_col_offset, int v_col_offset,
    int dst_row_stride,
    float eps) {
    (void)hidden;  // not needed for addressing (see .cuh: kept for interface
                    // completeness / future dst_row_stride != hidden cases)
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    // Precondition: HD must be even. Q/K/V slices are read/written via
    // __half2 (4-byte) vectorized loads at per-head byte offset
    // head*HD*sizeof(__half); that offset is only guaranteed 4-byte
    // aligned when HD is even (real ImageWAM: HD=128 for both the
    // backbone and ActionDiT, always even — see .cuh header). An odd
    // HD is out of scope here (not just numerically different from
    // the reference — it segfaults/misaligns), matching this
    // project's convention of validating kernels at real, fixed
    // production shapes rather than designing for arbitrary N.
    const int dim2 = HD >> 1;  // matches rms_norm_kernel<__half>'s own `dim >> 1`

    extern __shared__ float shared[];  // block_reduce_sum's own scratch, reused sequentially for Q then K

    const long qkv_row_base = (long)row * src_row_stride;
    const __half* q_src = qkv + qkv_row_base + q_col_offset + (long)head * HD;
    const __half* k_src = qkv + qkv_row_base + k_col_offset + (long)head * HD;
    const __half* v_src = qkv + qkv_row_base + v_col_offset + (long)head * HD;

    __half* q_dst = Q_out + (long)row * dst_row_stride + (long)head * HD;
    __half* k_dst = K_out + (long)row * dst_row_stride + (long)head * HD;
    __half* v_dst = V_out + (long)row * dst_row_stride + (long)head * HD;

    // RoPE table: shared across heads for a given row (rope_base =
    // seq_pos*HD in rope_apply_fp16_perhead_kernel, no head term).
    const __half* rope_row = rope_table + (long)row * HD;

    const __half2* q_src2 = reinterpret_cast<const __half2*>(q_src);
    const __half2* k_src2 = reinterpret_cast<const __half2*>(k_src);
    const __half2* qw2 = reinterpret_cast<const __half2*>(q_norm_weight);
    const __half2* kw2 = reinterpret_cast<const __half2*>(k_norm_weight);
    const __half2* rope2 = reinterpret_cast<const __half2*>(rope_row);
    __half2* q_dst2 = reinterpret_cast<__half2*>(q_dst);
    __half2* k_dst2 = reinterpret_cast<__half2*>(k_dst);

    // ---- pass 1: sum of squares (Q), matching rms_norm_kernel<__half> ----
    float q_local = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 v = q_src2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        q_local += v0 * v0 + v1 * v1;
    }
    const float q_rms = rsqrtf(block_reduce_sum(q_local, shared) / HD + eps);

    // block_reduce_sum's own internal syncs guarantee every thread has
    // WRITTEN its partial sum before any thread READS shared[0], but
    // there is no barrier AFTER that read. Without this explicit sync,
    // a fast thread can start WRITING K's own partial sums into the
    // same `shared[]` scratch (reused sequentially, see the .cuh
    // header) before a slower thread has finished READING Q's
    // shared[0] above -- a real race, not just a style nit (caught by
    // torch.equal failing on specific rows/columns during development,
    // not from static reasoning alone).
    __syncthreads();

    // ---- pass 1: sum of squares (K) ----
    float k_local = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 v = k_src2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        k_local += v0 * v0 + v1 * v1;
    }
    const float k_rms = rsqrtf(block_reduce_sum(k_local, shared) / HD + eps);

    // ---- pass 2: normalize (register-resident, same (x*rms)*w order
    // as rms_norm_kernel<__half>) then RoPE the just-normalized pair
    // in place (register round-trip through __half mirrors the
    // reference's real memory round-trip between its two separate
    // kernel launches bit-for-bit: __half2float(__float2half(v)) is
    // exactly what the RoPE kernel would read back from memory). ----
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 qv = q_src2[i];
        __half2 qw = qw2[i];
        float qn0 = __half2float(qv.x) * q_rms * __half2float(qw.x);
        float qn1 = __half2float(qv.y) * q_rms * __half2float(qw.y);
        const float q_normed0 = __half2float(__float2half(qn0));
        const float q_normed1 = __half2float(__float2half(qn1));

        __half2 kv = k_src2[i];
        __half2 kw = kw2[i];
        float kn0 = __half2float(kv.x) * k_rms * __half2float(kw.x);
        float kn1 = __half2float(kv.y) * k_rms * __half2float(kw.y);
        const float k_normed0 = __half2float(__float2half(kn0));
        const float k_normed1 = __half2float(__float2half(kn1));

        __half2 rp = rope2[i];
        const float c = __half2float(rp.x);
        const float s = __half2float(rp.y);

        q_dst2[i] = __halves2half2(
            __float2half(q_normed0 * c - q_normed1 * s),
            __float2half(q_normed1 * c + q_normed0 * s));
        k_dst2[i] = __halves2half2(
            __float2half(k_normed0 * c - k_normed1 * s),
            __float2half(k_normed1 * c + k_normed0 * s));
    }

    // ---- V: plain copy, no norm/rope — full HD width, elementwise ----
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        v_dst[i] = v_src[i];
    }
}

}  // namespace

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
    cudaStream_t stream) {
    dim3 grid(rows, NH);
    const size_t shared_bytes = 32 * sizeof(float);  // >= blockDim.x/32 floats
    qkv_split_norm_rope_fp16_kernel<<<grid, kBlockThreads, shared_bytes, stream>>>(
        qkv, q_norm_weight, k_norm_weight, rope_table,
        Q_out, K_out, V_out,
        NH, HD, hidden, src_row_stride,
        q_col_offset, k_col_offset, v_col_offset,
        dst_row_stride, eps);
}

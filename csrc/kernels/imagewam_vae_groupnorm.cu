// ================================================================
// FlashRT — ImageWAM VAE GroupNorm (+ optional SiLU), NHWC BF16.
// Contract: imagewam_vae_groupnorm.cuh.
//
// Statistics: each thread owns one 8-channel vector column and walks a
// strided subset of the block's pixels. Per pixel it folds the 4 or 8
// values that belong to one group into a (count, mean, M2) Welford
// state with Chan's merge; states are merged per group inside the block
// and across blocks in the finalize kernel.
// The build uses --use_fast_math, so the arithmetic that must be
// IEEE-rounded uses explicit _rn intrinsics.
// ================================================================

#include "imagewam_vae_groupnorm.cuh"

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocksPerSample = 256;

struct GroupNormPlan {
    int V;      // 8-channel vectors per pixel (C / 8)
    int P;      // pixel lanes per block (kThreads / V)
    int ppb;    // pixels per statistics block
    int B;      // statistics blocks per sample
    int Cg;     // channels per group
};

bool make_plan(int N, int HW, int C, int G, GroupNormPlan* plan) {
    if (N <= 0 || HW <= 0 || C <= 0 || G <= 0) return false;
    if (C % 8 != 0 || C % G != 0) return false;
    const int V = C / 8;
    if (kThreads % V != 0) return false;
    const int Cg = C / G;
    if (!(Cg == 4 || Cg % 8 == 0)) return false;
    const int P = kThreads / V;
    int ppb = (HW + kMaxBlocksPerSample - 1) / kMaxBlocksPerSample;
    if (ppb < 4 * P) ppb = 4 * P;
    ppb = (ppb + P - 1) / P * P;
    plan->V = V;
    plan->P = P;
    plan->ppb = ppb;
    plan->B = (HW + ppb - 1) / ppb;
    plan->Cg = Cg;
    return true;
}

size_t stats_floats(int N, int B, int G) { return static_cast<size_t>(N) * B * G * 3; }

__device__ __forceinline__ void chan_merge(float& na, float& ma, float& m2a, float nb, float mb, float m2b) {
    if (nb == 0.0f) return;
    const float n = __fadd_rn(na, nb);
    const float r = __fdiv_rn(nb, n);
    const float d = __fsub_rn(mb, ma);
    ma = __fmaf_rn(d, r, ma);
    m2a = __fadd_rn(__fadd_rn(m2a, m2b), __fmul_rn(__fmul_rn(__fmul_rn(d, d), na), r));
    na = n;
}

__device__ __forceinline__ void unpack8(const uint4& raw, float v[8]) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

// Welford state of a batch of K values folded into (na, ma, m2a).
template <int K>
__device__ __forceinline__ void fold_batch(const float* v, float& na, float& ma, float& m2a) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) s = __fadd_rn(s, v[i]);
    const float mb = __fmul_rn(s, 1.0f / K);
    float m2b = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        const float d = __fsub_rn(v[i], mb);
        m2b = __fmaf_rn(d, d, m2b);
    }
    chan_merge(na, ma, m2a, static_cast<float>(K), mb, m2b);
}

// v[k] = bf16(v[k] + bias[c0 + k]): torch's separate conv-bias add,
// with its BF16 rounding, folded into the load.
__device__ __forceinline__ void add_bias8(float v[8], const __nv_bfloat16* __restrict__ bias, int c0) {
    if (bias == nullptr) return;
    const uint4 raw = *reinterpret_cast<const uint4*>(bias + c0);
    float b[8];
    unpack8(raw, b);
    #pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(v[k], b[k])));
}

__global__ void __launch_bounds__(kThreads) gn_stats_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, float* __restrict__ part,
    int HW, int C, int G, int V, int P, int ppb, int B, int Cg)
{
    __shared__ float s_n[2][kThreads];
    __shared__ float s_mean[2][kThreads];
    __shared__ float s_m2[2][kThreads];

    const int b = blockIdx.x;
    const int n = blockIdx.y;
    const int t = threadIdx.x;
    const int cv = t % V;
    const int lane = t / V;
    const int p0 = b * ppb;
    const int p1 = min(p0 + ppb, HW);
    const __nv_bfloat16* xn = x + static_cast<size_t>(n) * HW * C + cv * 8;

    float cnt0 = 0.0f, mean0 = 0.0f, m20 = 0.0f;
    float cnt1 = 0.0f, mean1 = 0.0f, m21 = 0.0f;
    for (int p = p0 + lane; p < p1; p += P) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xn + static_cast<size_t>(p) * C);
        float v[8];
        unpack8(raw, v);
        add_bias8(v, bias, cv * 8);
        if (Cg == 4) {
            fold_batch<4>(v, cnt0, mean0, m20);
            fold_batch<4>(v + 4, cnt1, mean1, m21);
        } else {
            fold_batch<8>(v, cnt0, mean0, m20);
        }
    }
    s_n[0][t] = cnt0; s_mean[0][t] = mean0; s_m2[0][t] = m20;
    s_n[1][t] = cnt1; s_mean[1][t] = mean1; s_m2[1][t] = m21;
    __syncthreads();

    if (t < G) {
        const int g = t;
        float cn = 0.0f, mn = 0.0f, mm = 0.0f;
        if (Cg == 4) {
            const int cvg = g >> 1;
            const int slot = g & 1;
            for (int l = 0; l < P; ++l) {
                const int src = l * V + cvg;
                chan_merge(cn, mn, mm, s_n[slot][src], s_mean[slot][src], s_m2[slot][src]);
            }
        } else {
            const int vecs = Cg / 8;
            for (int k = 0; k < vecs; ++k) {
                const int cvg = g * vecs + k;
                for (int l = 0; l < P; ++l) {
                    const int src = l * V + cvg;
                    chan_merge(cn, mn, mm, s_n[0][src], s_mean[0][src], s_m2[0][src]);
                }
            }
        }
        float* dst = part + ((static_cast<size_t>(n) * B + b) * G + g) * 3;
        dst[0] = cn;
        dst[1] = mn;
        dst[2] = mm;
    }
}

// One block per (group, sample): thread b takes partial state b, the
// states are Chan-merged with warp shuffles and then across warps, and
// the group's channels get torch's ComputeFusedParamsCUDAKernel values:
// a = rstd * gamma, b = -a * mean + beta.
__global__ void __launch_bounds__(kThreads) gn_finalize_kernel(
    const float* __restrict__ part, const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta, float2* __restrict__ ab,
    int C, int G, int B, int Cg, float eps)
{
    __shared__ float s_n[kThreads / 32], s_mean[kThreads / 32], s_m2[kThreads / 32];
    __shared__ float s_stat[2];
    const int g = blockIdx.x;
    const int n = blockIdx.y;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    float cn = 0.0f, mn = 0.0f, mm = 0.0f;
    if (t < B) {
        const float* s = part + ((static_cast<size_t>(n) * B + t) * G + g) * 3;
        cn = s[0];
        mn = s[1];
        mm = s[2];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float on = __shfl_down_sync(0xffffffffu, cn, off);
        const float om = __shfl_down_sync(0xffffffffu, mn, off);
        const float o2 = __shfl_down_sync(0xffffffffu, mm, off);
        chan_merge(cn, mn, mm, on, om, o2);
    }
    if (lane == 0) {
        s_n[warp] = cn;
        s_mean[warp] = mn;
        s_m2[warp] = mm;
    }
    __syncthreads();
    if (t == 0) {
        float tn = 0.0f, tm = 0.0f, t2 = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) chan_merge(tn, tm, t2, s_n[w], s_mean[w], s_m2[w]);
        float var = tn > 0.0f ? __fdiv_rn(t2, tn) : 0.0f;
        if (var < 0.0f) var = 0.0f;
        s_stat[0] = tm;
        s_stat[1] = __frsqrt_rn(__fadd_rn(var, eps));
    }
    __syncthreads();
    const float mean_f = s_stat[0];
    const float rstd = s_stat[1];
    for (int k = t; k < Cg; k += kThreads) {
        const int c = g * Cg + k;
        const float a = __fmul_rn(rstd, __bfloat162float(gamma[c]));
        const float bb = __fmaf_rn(-a, mean_f, __bfloat162float(beta[c]));
        ab[static_cast<size_t>(n) * C + c] = make_float2(a, bb);
    }
}

__global__ void __launch_bounds__(kThreads) gn_apply_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* __restrict__ bias, const float2* __restrict__ ab,
    __nv_bfloat16* y, long long total_vec, long long hw_vec, int V, int C, int apply_silu)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total_vec; i += stride) {
        const int n = static_cast<int>(i / hw_vec);
        const int c0 = static_cast<int>(i % V) * 8;
        const uint4 raw = *reinterpret_cast<const uint4*>(x + i * 8);
        float v[8];
        unpack8(raw, v);
        add_bias8(v, bias, c0);
        const float2* abn = ab + static_cast<size_t>(n) * C + c0;
        uint4 packed;
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&packed);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float2 p = abn[k];
            __nv_bfloat16 r = __float2bfloat16_rn(__fmaf_rn(v[k], p.x, p.y));
            if (apply_silu) {
                const float nf = __bfloat162float(r);
                const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-nf)));
                r = __float2bfloat16_rn(__fmul_rn(nf, __bfloat162float(__float2bfloat16_rn(sg))));
            }
            o[k] = r;
        }
        *reinterpret_cast<uint4*>(y + i * 8) = packed;
    }
}

}  // namespace

size_t imagewam_groupnorm_nhwc_workspace_bytes(int N, int HW, int C, int G) {
    GroupNormPlan plan;
    if (!make_plan(N, HW, C, G, &plan)) return 0;
    const size_t stats = stats_floats(N, plan.B, G) * sizeof(float);
    const size_t stats_aligned = (stats + 15) / 16 * 16;
    return stats_aligned + static_cast<size_t>(N) * C * sizeof(float2);
}

int imagewam_groupnorm_nhwc_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias, const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* y, void* workspace, size_t workspace_bytes,
    int N, int HW, int C, int G, float eps, int apply_silu, cudaStream_t stream)
{
    GroupNormPlan plan;
    if (!make_plan(N, HW, C, G, &plan)) return -1;
    if (x == nullptr || y == nullptr || gamma == nullptr || beta == nullptr || workspace == nullptr) return -2;
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(y) |
         reinterpret_cast<uintptr_t>(bias) | reinterpret_cast<uintptr_t>(workspace)) % 16 != 0) return -3;
    if (workspace_bytes < imagewam_groupnorm_nhwc_workspace_bytes(N, HW, C, G)) return -4;

    float* part = static_cast<float*>(workspace);
    const size_t stats = stats_floats(N, plan.B, G) * sizeof(float);
    float2* ab = reinterpret_cast<float2*>(static_cast<char*>(workspace) + (stats + 15) / 16 * 16);

    gn_stats_kernel<<<dim3(plan.B, N), kThreads, 0, stream>>>(
        x, bias, part, HW, C, G, plan.V, plan.P, plan.ppb, plan.B, plan.Cg);
    gn_finalize_kernel<<<dim3(G, N), kThreads, 0, stream>>>(part, gamma, beta, ab, C, G, plan.B, plan.Cg, eps);
    const long long total_vec = static_cast<long long>(N) * HW * plan.V;
    const long long blocks = (total_vec + kThreads - 1) / kThreads;
    gn_apply_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        x, bias, ab, y, total_vec, static_cast<long long>(HW) * plan.V, plan.V, C, apply_silu);
    return static_cast<int>(cudaGetLastError());
}

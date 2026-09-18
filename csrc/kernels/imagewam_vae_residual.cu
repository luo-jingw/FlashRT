// ================================================================
// FlashRT — ImageWAM VAE ResnetBlock tail. Contract:
// imagewam_vae_residual.cuh. Every add is an explicit _rn intrinsic
// followed by BF16 rounding, matching torch's separate BF16 adds under
// --use_fast_math.
// ================================================================

#include "imagewam_vae_residual.cuh"

#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float v[8]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ float add_bf16(float a, float b) {
    return __bfloat162float(__float2bfloat16_rn(__fadd_rn(a, b)));
}

__global__ void __launch_bounds__(kThreads) bias_residual_kernel(
    const __nv_bfloat16* h, const __nv_bfloat16* __restrict__ h_bias,
    const __nv_bfloat16* res, const __nv_bfloat16* __restrict__ res_bias,
    __nv_bfloat16* y, long long total_vec, int V)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total_vec; i += stride) {
        const int c0 = static_cast<int>(i % V) * 8;
        float hv[8], hb[8], rv[8];
        load8(h + i * 8, hv);
        load8(h_bias + c0, hb);
        load8(res + i * 8, rv);
        if (res_bias != nullptr) {
            float rb[8];
            load8(res_bias + c0, rb);
            #pragma unroll
            for (int k = 0; k < 8; ++k) rv[k] = add_bf16(rv[k], rb[k]);
        }
        uint4 packed;
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&packed);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            o[k] = __float2bfloat16_rn(__fadd_rn(rv[k], add_bf16(hv[k], hb[k])));
        }
        *reinterpret_cast<uint4*>(y + i * 8) = packed;
    }
}

}  // namespace

int imagewam_bias_residual_nhwc_bf16(
    const __nv_bfloat16* h, const __nv_bfloat16* h_bias,
    const __nv_bfloat16* res, const __nv_bfloat16* res_bias,
    __nv_bfloat16* y, long long rows, int C, cudaStream_t stream)
{
    if (h == nullptr || h_bias == nullptr || res == nullptr || y == nullptr) return -1;
    if (rows <= 0 || C <= 0 || C % 8 != 0) return -2;
    if ((reinterpret_cast<uintptr_t>(h) | reinterpret_cast<uintptr_t>(h_bias) | reinterpret_cast<uintptr_t>(res) |
         reinterpret_cast<uintptr_t>(res_bias) | reinterpret_cast<uintptr_t>(y)) % 16 != 0) return -3;
    const int V = C / 8;
    const long long total_vec = rows * V;
    const long long blocks = (total_vec + kThreads - 1) / kThreads;
    bias_residual_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        h, h_bias, res, res_bias, y, total_vec, V);
    return static_cast<int>(cudaGetLastError());
}

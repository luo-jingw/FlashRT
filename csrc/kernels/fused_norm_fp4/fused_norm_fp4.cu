// ============================================================================
//  Fused gated-residual + AdaLayerNorm(no-affine) + NVFP4 quantize + CUTLASS
//  SFA write. See fused_norm_fp4.cuh for the two-kernel sequence this
//  replaces and the numerical-ground-truth files this was checked against:
//
//    - residual update math: csrc/kernels/decoder_fused.cu's
//      gate_res_bf16res_kernel / gate_res_fp16_kernel (verbatim expression)
//    - LayerNorm mean/var reduction: csrc/kernels/norm.cu's
//      ada_layer_norm_bf16in_fp16out_kernel / ada_layer_norm_fp16_kernel
//      (verbatim shfl_xor butterfly reduction -- NOT common.cuh's
//      block_reduce_sum, which uses a different shfl_down pattern and would
//      NOT reproduce the same summation order / float rounding)
//    - FP4 e2m1 pack + UE4M3 SFA-block scale + CUTLASS tile-interleaved
//      layout write: csrc/quantize/quantize_fp4_sfa.cu's
//      kernel_quantize_fp4_sfa (verbatim per-16-block formulas)
//
//  The modulated value is rounded through fp16 in registers
//  (to_f32<half>(from_f32<half>(v))) before the amax/quantize step, matching
//  the established idiom in csrc/fused_fp4/layer_norm_fp4_sfa.cu /
//  dit_norm_fp4_sfa.cu ("rounded through fp16/bf16 before quantization, so
//  the output is bit-identical to running the norm kernel then
//  quantize_fp4_dynamic_sfa_fp16") -- ada_layer_norm_bf16in_fp16out_kernel's
//  own `out` buffer is real fp16 storage, so this reproduces that exact
//  round-trip without materializing the buffer.
//
//  Standalone/additive: does not modify any file it is checked against.
// ============================================================================
#include "fused_norm_fp4.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_norm_fp4 {

#if FV_HAVE_CUTLASS

using CfgFN = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// ── dtype-generic helpers, duplicated locally (same convention
//    quantize_fp4_sfa.cu itself uses: "duplicated locally to stay additive
//    -- not linking against [other kernel files] so we don't risk ODR
//    issues") ──
template<typename T> __device__ __forceinline__ float fn_to_f32(T x);
template<> __device__ __forceinline__ float fn_to_f32<__half>(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float fn_to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T> __device__ __forceinline__ T fn_from_f32(float x);
template<> __device__ __forceinline__ __half fn_from_f32<__half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 fn_from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template<typename T> struct FnPacked2;
template<> struct FnPacked2<__half> { using type = __half2; };
template<> struct FnPacked2<__nv_bfloat16> { using type = __nv_bfloat162; };

__device__ __forceinline__ uint8_t fn_fp32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

// One CTA per row, 256 threads (matches ada_layer_norm_bf16in_fp16out's own
// <<<seq_len, 256, 256*sizeof(float)>>> launch exactly -- required for the
// mean/var shfl_xor reduction below to reproduce the identical summation
// order/rounding as that kernel).
template <typename ResT, class LayoutSF>
__global__ void gate_res_ada_layer_norm_fp4_sfa_kernel(
    ResT* __restrict__ residual,
    const __half* __restrict__ gemm_out,
    const __half* __restrict__ gate,
    const __half* __restrict__ scale,
    const __half* __restrict__ shift,
    const __half* __restrict__ inv_s,   // nullable
    uint8_t* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int dim, float eps) {
    using T2 = typename FnPacked2<ResT>::type;
    const int row = blockIdx.x;
    ResT* res_row = residual + static_cast<long>(row) * dim;
    const __half* gemm_row = gemm_out + static_cast<long>(row) * dim;

    // ── Stage A: gated residual update -- verbatim per-element expression
    // from gate_res_bf16res_kernel / gate_res_fp16_kernel (decoder_fused.cu).
    // No cross-thread reduction here, so the thread/block mapping cannot
    // change the per-element result -- only the exact expression text (kept
    // identical) and compile flags matter for bit-exactness. ──
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float r = fn_to_f32<ResT>(res_row[i]) +
                  fn_to_f32<__half>(gemm_row[i]) * fn_to_f32<__half>(gate[i]);
        res_row[i] = fn_from_f32<ResT>(r);
    }
    __syncthreads();  // residual fully updated before Stage B reads it back

    // ── Stage B/C: LayerNorm(no affine) mean/var -- verbatim reduction
    // pattern from ada_layer_norm_bf16in_fp16out_kernel / ada_layer_norm_
    // fp16_kernel (norm.cu): shfl_xor butterfly, NOT common.cuh's
    // block_reduce_sum (shfl_down -- a different summation order). ──
    const T2* res2 = reinterpret_cast<const T2*>(res_row);
    const int dim2 = dim >> 1;
    extern __shared__ float shared[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 v = res2[i];
        local_sum += fn_to_f32<ResT>(v.x) + fn_to_f32<ResT>(v.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 v = res2[i];
        float d0 = fn_to_f32<ResT>(v.x) - mean, d1 = fn_to_f32<ResT>(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    // ── Stage D: modulate (verbatim ada_layer_norm formula) + fp16 round-
    // trip + FP4 e2m1 pack + UE4M3 SFA scale (verbatim kernel_quantize_
    // fp4_sfa formulas) + CUTLASS tile-interleaved SFA byte write. ──
    const int n_blocks = dim >> 4;
    for (int blk = threadIdx.x; blk < n_blocks; blk += blockDim.x) {
        const int base = blk * 16;
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float n0 = (fn_to_f32<ResT>(res_row[base + i]) - mean) * inv_std;
            float v0 = n0 * (1.0f + fn_to_f32<__half>(scale[base + i])) +
                       fn_to_f32<__half>(shift[base + i]);
            if (inv_s != nullptr) v0 = v0 * fn_to_f32<__half>(inv_s[base + i]);
            vals[i] = fn_to_f32<__half>(fn_from_f32<__half>(v0));  // fp16 round-trip
            float a = fabsf(vals[i]);
            if (a > amax) amax = a;
        }

        float desired = amax / 6.f;
        if (desired < 1e-12f) desired = 1e-12f;
        __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
        float bs_dq = static_cast<float>(bs_q);
        int sfa_off = layout(row, base, 0);
        dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

        const int out_base = row * (dim / 2) + blk * 8;
        const float inv_bs = 1.f / bs_dq;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            float v_lo = vals[2 * p] * inv_bs;
            float v_hi = vals[2 * p + 1] * inv_bs;
            uint8_t lo = fn_fp32_to_e2m1(v_lo);
            uint8_t hi = fn_fp32_to_e2m1(v_hi);
            dst_packed[out_base + p] = lo | (hi << 4);
        }
    }
}

template <typename ResT>
static int launch(void* residual, const void* gemm_out, const void* gate,
                   const void* scale, const void* shift, const void* inv_s,
                   void* dst_packed, void* dst_sfa,
                   int seq_len, int dim, float eps, cudaStream_t stream) {
    if (dim % 16 != 0 || dim % 2 != 0) return -1;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);  // SFA shape (is_sfb=false)
    auto layout = CfgFN::tile_atom_to_shape_SFA(shape);
    gate_res_ada_layer_norm_fp4_sfa_kernel<ResT><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<ResT*>(residual),
        reinterpret_cast<const __half*>(gemm_out),
        reinterpret_cast<const __half*>(gate),
        reinterpret_cast<const __half*>(scale),
        reinterpret_cast<const __half*>(shift),
        reinterpret_cast<const __half*>(inv_s),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, dim, eps);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

#endif  // FV_HAVE_CUTLASS

int gate_res_ada_layer_norm_fp4_sfa_bf16res(
    void* residual_bf16, const void* gemm_out_fp16, const void* gate_fp16,
    const void* scale_fp16, const void* shift_fp16, const void* inv_s_fp16,
    void* dst_packed, void* dst_sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    return launch<__nv_bfloat16>(residual_bf16, gemm_out_fp16, gate_fp16,
                                  scale_fp16, shift_fp16, inv_s_fp16,
                                  dst_packed, dst_sfa, seq_len, dim, eps, stream);
#else
    (void)residual_bf16; (void)gemm_out_fp16; (void)gate_fp16; (void)scale_fp16;
    (void)shift_fp16; (void)inv_s_fp16; (void)dst_packed; (void)dst_sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

int gate_res_ada_layer_norm_fp4_sfa_fp16res(
    void* residual_fp16, const void* gemm_out_fp16, const void* gate_fp16,
    const void* scale_fp16, const void* shift_fp16, const void* inv_s_fp16,
    void* dst_packed, void* dst_sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    return launch<__half>(residual_fp16, gemm_out_fp16, gate_fp16,
                           scale_fp16, shift_fp16, inv_s_fp16,
                           dst_packed, dst_sfa, seq_len, dim, eps, stream);
#else
    (void)residual_fp16; (void)gemm_out_fp16; (void)gate_fp16; (void)scale_fp16;
    (void)shift_fp16; (void)inv_s_fp16; (void)dst_packed; (void)dst_sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

}  // namespace fused_norm_fp4
}  // namespace flash_rt

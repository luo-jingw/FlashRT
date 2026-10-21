// Prototype NVFP4 activation-quantization kernels for the ImageWAM ActionDiT NVFP4 path (SM120).
//
// Production chain per NVFP4 GEMM input (Nvfp4LinearSm120):
//   cast_fp16_to_bf16  ->  quantize_bf16_to_nvfp4_swizzled  (one block per row)
// and, for inputs produced by the fused residual + AdaLN kernel, that kernel
// (gate_res_ada_layer_norm_fp16) runs first and writes the fp16 `modded` row.
//
// proto_quant_fp16_nvfp4: one thread per 16-element block (grid over all
//   rows x blocks), reads fp16, rounds through bf16 in registers (the cast the
//   production chain does), then the production quantization math
//   (amax/6 -> UE4M3 ceil -> 1/scale -> E2M1). Bit-identical to
//   cast_fp16_to_bf16 + quantize_bf16_to_nvfp4_swizzled.
// proto_gate_res_ada_ln_nvfp4: gate_res_ada_layer_norm_kernel<__half> with the
//   same passes 1-2; pass 3 walks 16-element blocks, writes the fp16 `modded`
//   row as before and the NVFP4 packed row + swizzled scale factors computed
//   from it (same rounding chain as above). Bit-identical to the three
//   production kernels.
// Build with the production flags (--ftz=true --prec-div=false --prec-sqrt=false).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "nvfp4_convert.cuh"

__device__ __forceinline__ float through_bf16(__half h) {
    return __bfloat162float(__float2bfloat16(__half2float(h)));
}

__device__ __forceinline__ void quant_block16(const float q[16], int row, int b, int n_col_blocks,
                                              uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ sf) {
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(q[j]));
    uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    int rb = row / 128, ri = row % 128, cb = b / 4, ci = b % 4;
    sf[(rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + ci] = ue;
    float scale = ue4m3_to_float(ue);
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    uint8_t packed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint8_t lo = float_to_fp4_e2m1(q[2 * j] * inv);
        uint8_t hi = float_to_fp4_e2m1(q[2 * j + 1] * inv);
        packed[j] = (hi << 4) | (lo & 0x0F);
    }
    *reinterpret_cast<uint2*>(row_fp4 + b * 8) = *reinterpret_cast<uint2*>(packed);
}

__global__ void proto_quant_fp16_nvfp4_kernel(const __half* __restrict__ in, uint8_t* __restrict__ fp4,
                                              uint8_t* __restrict__ sf, int rows, int cols, int n_col_blocks) {
    int nb = cols / 16;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * nb) return;
    int row = t / nb, b = t % nb;
    const int4* src = reinterpret_cast<const int4*>(in + (size_t)row * cols + b * 16);
    int4 v[2] = {src[0], src[1]};
    const __half* h = reinterpret_cast<const __half*>(v);
    float q[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) q[j] = through_bf16(h[j]);
    quant_block16(q, row, b, n_col_blocks, fp4 + (size_t)row * cols / 2, sf);
}

__device__ __forceinline__ float round_through_fp16(float x) { return __half2float(__float2half(x)); }

__global__ void proto_gate_res_ada_ln_nvfp4_kernel(const __half* __restrict__ proj, const float* __restrict__ gate,
                                                   __half* __restrict__ residual, const float* __restrict__ scale,
                                                   const float* __restrict__ shift, __half* __restrict__ out,
                                                   uint8_t* __restrict__ fp4, uint8_t* __restrict__ sf,
                                                   int dim, float eps, int n_col_blocks) {
    int row = blockIdx.x;
    __half2* res2 = reinterpret_cast<__half2*>(residual + (size_t)row * dim);
    const __half2* p2 = reinterpret_cast<const __half2*>(proj + (size_t)row * dim);
    int dim2 = dim >> 1;
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 rv = res2[i];
        __half2 pv = p2[i];
        float g0 = round_through_fp16(gate[2 * i]);
        float g1 = round_through_fp16(gate[2 * i + 1]);
        float r0 = __half2float(rv.x) + __half2float(pv.x) * g0;
        float r1 = __half2float(rv.y) + __half2float(pv.y) * g1;
        __half s0 = __float2half(r0), s1 = __float2half(r1);
        res2[i] = __halves2half2(s0, s1);
        local_sum += __half2float(s0) + __half2float(s1);
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
    __syncthreads();
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 v = res2[i];
        float d0 = __half2float(v.x) - mean, d1 = __half2float(v.y) - mean;
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
    // Pass 3 (block-wise): every element is written by the thread that read it
    // in passes 1-2 only when blockDim covers the row; read back from the
    // residual row in global memory (written above, visible after the barrier).
    __syncthreads();
    const __half* res = residual + (size_t)row * dim;
    __half* orow = out + (size_t)row * dim;
    uint8_t* row_fp4 = fp4 + (size_t)row * dim / 2;
    for (int b = threadIdx.x; b < dim / 16; b += blockDim.x) {
        float q[16];
        __half o16[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int c = b * 16 + j;
            float n = (__half2float(res[c]) - mean) * inv_std;
            float v = n * (1.0f + round_through_fp16(scale[c])) + round_through_fp16(shift[c]);
            o16[j] = __float2half(v);
            q[j] = through_bf16(o16[j]);
        }
        int4* dst = reinterpret_cast<int4*>(orow + b * 16);
        dst[0] = reinterpret_cast<int4*>(o16)[0];
        dst[1] = reinterpret_cast<int4*>(o16)[1];
        quant_block16(q, row, b, n_col_blocks, row_fp4, sf);
    }
}

extern "C" {
int proto_quant_fp16_nvfp4(const void* in, void* fp4, void* sf, int rows, int cols, void* stream) {
    if (cols % 16) return -1;
    int nb = cols / 16, n_col_blocks = (nb + 3) / 4;
    int total = rows * nb, threads = 128;
    proto_quant_fp16_nvfp4_kernel<<<(total + threads - 1) / threads, threads, 0, (cudaStream_t)stream>>>(
        (const __half*)in, (uint8_t*)fp4, (uint8_t*)sf, rows, cols, n_col_blocks);
    return (int)cudaGetLastError();
}
int proto_gate_res_ada_ln_nvfp4(const void* proj, const void* gate, void* residual, const void* scale,
                                const void* shift, void* out, void* fp4, void* sf, int rows, int dim, float eps,
                                void* stream) {
    if (dim % 16) return -1;
    int nb = dim / 16, n_col_blocks = (nb + 3) / 4;
    proto_gate_res_ada_ln_nvfp4_kernel<<<rows, 256, 256 * sizeof(float), (cudaStream_t)stream>>>(
        (const __half*)proj, (const float*)gate, (__half*)residual, (const float*)scale, (const float*)shift,
        (__half*)out, (uint8_t*)fp4, (uint8_t*)sf, dim, eps, n_col_blocks);
    return (int)cudaGetLastError();
}
}

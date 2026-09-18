// ================================================================
// FlashRT — Cross-layer fusion kernels (dtype-generic)
// Fused gate*residual + AdaRMSNorm -> FP8 (decoder optimization)
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "fusion.cuh"
#include "common.cuh"

#include <stdexcept>
#include <string>

// ── Fused Gate*Residual + AdaRMSNorm + Style -> FP8 ──
// Pass 1: residual += x * gate; sum_sq += residual^2
// Reduce: rms = rsqrt(sum_sq / dim + eps)
// Pass 2: out_fp8 = clamp((norm(residual) * (1+scale) + shift) * inv_scale)
template<typename T>
__global__ void gate_residual_ada_norm_fp8_kernel(
    T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out,
    T* __restrict__ gate_out,
    int dim, float eps,
    const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    __nv_fp8_e4m3* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    // Pass 1: residual += x * gate, compute sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    // Pass 2: AdaRMSNorm with style -> FP8
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate_out2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_fp8_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, const __half*,
    __nv_fp8_e4m3*, __half*, int, float, const float*))
FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_fp8_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_fp8_e4m3*, __nv_bfloat16*, int, float, const float*))
void gate_residual_ada_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 const float* d_scale, cudaStream_t stream) {
    gate_residual_ada_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
}
void gate_residual_ada_norm_fp8_fp16(__half* residual, const __half* x,
                                      const __half* gate, const __half* weight,
                                      const __half* style,
                                      __nv_fp8_e4m3* out, __half* gate_out,
                                      int seq_len, int dim, float eps,
                                      const float* d_scale, cudaStream_t stream) {
    gate_residual_ada_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
}

// ── Fused Gate*Residual + AdaRMSNorm + Style -> FP16 ──
// Same math as gate_mul_residual_fp16 + ada_rms_norm_style_fp16, but
// keeps the residual update and RMS reduction in one launch.
__global__ void gate_residual_ada_norm_fp16_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ gate,
    const __half* __restrict__ weight,
    const __half* __restrict__ style,
    __half* __restrict__ out,
    __half* __restrict__ gate_out,
    int dim, float eps) {
    using T2 = typename packed2<__half>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const __half* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        __half rh0 = from_f32<__half>(r0);
        __half rh1 = from_f32<__half>(r1);
        res2[i] = make_packed2<__half>(rh0, rh1);
        float rr0 = to_f32(rh0);
        float rr1 = to_f32(rh1);
        local_sum += rr0 * rr0 + rr1 * rr1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<__half>(
            from_f32<__half>(n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)),
            from_f32<__half>(n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)));
        gate_out2[i] = gv;
    }
}

void gate_residual_ada_norm_fp16(__half* residual, const __half* x,
                                  const __half* gate, const __half* weight,
                                  const __half* style,
                                  __half* out, __half* gate_out,
                                  int seq_len, int dim, float eps,
                                  cudaStream_t stream) {
    gate_residual_ada_norm_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps);
}

template<typename T>
__global__ void gate_residual_ada_norm_int8_kernel(
    T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    int8_t* __restrict__ out,
    T* __restrict__ gate_out,
    int dim, float eps,
    float* __restrict__ d_scales) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    int8_t* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    float local_amax = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x);
        float val1 = n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y);
        local_amax = fmaxf(local_amax, fabsf(val0));
        local_amax = fmaxf(local_amax, fabsf(val1));
    }
    float amax = block_reduce_max(local_amax, shared);
    __shared__ float scale_s;
    if (threadIdx.x == 0) {
        float s = fmaxf(amax / 127.0f, 1e-10f);
        d_scales[row] = s;
        scale_s = s;
    }
    __syncthreads();
    float inv_scale = 1.0f / scale_s;

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        int q0 = __float2int_rn(val0);
        int q1 = __float2int_rn(val1);
        out_row[2 * i] = static_cast<int8_t>((q0 < -127) ? -127 : ((q0 > 127) ? 127 : q0));
        out_row[2 * i + 1] = static_cast<int8_t>((q1 < -127) ? -127 : ((q1 > 127) ? 127 : q1));
        gate_out2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_int8_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, const __half*,
    int8_t*, __half*, int, float, float*))
FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_int8_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    int8_t*, __nv_bfloat16*, int, float, float*))
void gate_residual_ada_norm_int8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 int8_t* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 float* d_scales, cudaStream_t stream) {
    gate_residual_ada_norm_int8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scales);
}

// ── Fused gated residual + next AdaLayerNorm (ImageWAM / FLUX.2 DiT) ──
// One block per row. Same math, bit for bit, as the unfused pair
//   gate_res_{bf16res,fp16}(proj, gate_rows, residual)       (decoder_fused.cu)
//   ada_layer_norm_{bf16in_fp16out,fp16}(residual, scale, shift, out)  (norm.cu)
// with the FP16 modulation vectors those kernels take:
//   residual[r,c] = RES(float(residual[r,c]) + float(proj[r,c]) * h(gate[c]))
//   out[r,:]      = fp16(LN_no_affine(residual[r,:]) * (1 + h(scale)) + h(shift))
// where h(x) = float(fp16(x)). gate/scale/shift are the (dim,) FP32 rows of
// the AdaLN modulation output; rounding them to FP16 here reproduces the
// fp32->fp16 cast the unfused path does before its kernels. The LayerNorm
// statistics use the STORED (rounded) residual and the same thread-strided
// accumulation and block reduction as ada_layer_norm_*, so the result is
// identical. `dim` must be even; launch with 256 threads.
// `out == nullptr`: residual update only (no AdaLN follows, e.g. the
// backbone's last layer); `scale`/`shift` are then not read.
__device__ __forceinline__ float round_through_fp16(float x) {
    return __half2float(__float2half(x));
}

template<typename ResT>
__global__ void gate_res_ada_layer_norm_kernel(
    const __half* __restrict__ proj,
    const float* __restrict__ gate,
    ResT* __restrict__ residual,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    __half* __restrict__ out,
    int dim, float eps) {
    using R2 = typename packed2<ResT>::type;
    int row = blockIdx.x;
    R2* res2 = reinterpret_cast<R2*>(residual + (size_t)row * dim);
    const __half2* p2 = reinterpret_cast<const __half2*>(proj + (size_t)row * dim);
    __half2* out2 = reinterpret_cast<__half2*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    // Pass 1: residual update, row sum of the stored residual.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        R2 rv = res2[i];
        __half2 pv = p2[i];
        float g0 = round_through_fp16(gate[2 * i]);
        float g1 = round_through_fp16(gate[2 * i + 1]);
        float r0 = to_f32(rv.x) + __half2float(pv.x) * g0;
        float r1 = to_f32(rv.y) + __half2float(pv.y) * g1;
        ResT s0 = from_f32<ResT>(r0);
        ResT s1 = from_f32<ResT>(r1);
        res2[i] = make_packed2<ResT>(s0, s1);
        local_sum += to_f32(s0) + to_f32(s1);
    }
    if (out == nullptr) return;  // uniform across the block
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float mean = shared[0] / dim;
    __syncthreads();  // every thread has read `mean` before shared[] is reused

    // Pass 2: variance (each thread re-reads only the elements it wrote).
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        R2 v = res2[i];
        float d0 = to_f32(v.x) - mean, d1 = to_f32(v.y) - mean;
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

    // Pass 3: normalize + modulate.
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        R2 xv = res2[i];
        float n0 = (to_f32(xv.x) - mean) * inv_std;
        float n1 = (to_f32(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + round_through_fp16(scale[2 * i])) + round_through_fp16(shift[2 * i]);
        float v1 = n1 * (1.0f + round_through_fp16(scale[2 * i + 1])) + round_through_fp16(shift[2 * i + 1]);
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void gate_res_ada_layer_norm_kernel<__half>(
    const __half*, const float*, __half*, const float*, const float*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void gate_res_ada_layer_norm_kernel<__nv_bfloat16>(
    const __half*, const float*, __nv_bfloat16*, const float*, const float*, __half*, int, float))

// The kernel walks each row as packed pairs (__half2 / __nv_bfloat162),
// so an odd `dim` would misalign every other row.
static void check_gate_res_ada_layer_norm_shape(int rows, int dim) {
    if (rows <= 0 || dim <= 0 || (dim & 1)) {
        throw std::invalid_argument("gate_res_ada_layer_norm: need rows > 0 and an even dim > 0, got rows=" +
                                    std::to_string(rows) + " dim=" + std::to_string(dim));
    }
}

void gate_res_ada_layer_norm_bf16res(const __half* proj, const float* gate,
                                     __nv_bfloat16* residual,
                                     const float* scale, const float* shift, __half* out,
                                     int rows, int dim, float eps, cudaStream_t stream) {
    check_gate_res_ada_layer_norm_shape(rows, dim);
    gate_res_ada_layer_norm_kernel<__nv_bfloat16><<<rows, 256, 256 * sizeof(float), stream>>>(
        proj, gate, residual, scale, shift, out, dim, eps);
}

void gate_res_ada_layer_norm_fp16(const __half* proj, const float* gate,
                                  __half* residual,
                                  const float* scale, const float* shift, __half* out,
                                  int rows, int dim, float eps, cudaStream_t stream) {
    check_gate_res_ada_layer_norm_shape(rows, dim);
    gate_res_ada_layer_norm_kernel<__half><<<rows, 256, 256 * sizeof(float), stream>>>(
        proj, gate, residual, scale, shift, out, dim, eps);
}

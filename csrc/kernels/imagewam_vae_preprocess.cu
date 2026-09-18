// ================================================================
// FlashRT — ImageWAM VAE input preprocessing kernel.
// See imagewam_vae_preprocess.cuh for the contract of each mode.
//
// The build compiles with --use_fast_math, so every float operation
// that must match torch bit-for-bit uses an explicit round-to-nearest
// intrinsic (__fdiv_rn / __fmul_rn / __fadd_rn), which fast-math does
// not rewrite or contract.
// ================================================================

#include "imagewam_vae_preprocess.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kPilPrecisionBits = 22;  // PIL Resample.c PRECISION_BITS (32 - 8 - 2)

__device__ __forceinline__ int pil_clip8(int acc) {
    int v = acc >> kPilPrecisionBits;
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

// One horizontal-pass sample of PIL's resample: row `row` of the input,
// resized column `rx`, channel `c`. Integer math identical to
// ImagingResampleHorizontal_8bpc.
__device__ __forceinline__ int pil_horizontal(
    const uint8_t* __restrict__ view, int in_w, int row, int rx, int c,
    const int* __restrict__ h_bounds, const int* __restrict__ h_coeffs, int h_ksize)
{
    if (h_bounds == nullptr) {
        return view[(row * in_w + rx) * 3 + c];
    }
    const int xmin = h_bounds[rx * 2 + 0];
    const int xlen = h_bounds[rx * 2 + 1];
    const int* k = h_coeffs + rx * h_ksize;
    int acc = 1 << (kPilPrecisionBits - 1);
    const uint8_t* src = view + (row * in_w + xmin) * 3 + c;
    for (int t = 0; t < xlen; ++t) {
        acc += static_cast<int>(src[t * 3]) * k[t];
    }
    return pil_clip8(acc);
}

__global__ void imagewam_vae_preprocess_kernel(
    const uint8_t* __restrict__ view,
    const __nv_bfloat16* __restrict__ lut,
    __nv_bfloat16* __restrict__ out,
    int in_h, int in_w, int out_h, int out_w, int out_total_w, int col_offset,
    int mode,
    const int* __restrict__ h_bounds, const int* __restrict__ h_coeffs, int h_ksize, int crop_left,
    const int* __restrict__ v_bounds, const int* __restrict__ v_coeffs, int v_ksize, int crop_top,
    float inv255)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_h * out_w) return;
    const int oy = idx / out_w;
    const int ox = idx - oy * out_w;
    const long long plane = static_cast<long long>(out_h) * out_total_w;
    const long long dst = static_cast<long long>(oy) * out_total_w + col_offset + ox;

    if (mode == kImageWAMVaeResizeIdentity) {
        const uint8_t* src = view + (oy * in_w + ox) * 3;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            out[c * plane + dst] = lut[src[c]];
        }
        return;
    }

    if (mode == kImageWAMVaeResizeArea) {
        // torch adaptive_avg_pool2d window: floor(o*in/out) .. ceil((o+1)*in/out)
        const int hs = (oy * in_h) / out_h;
        const int he = ((oy + 1) * in_h + out_h - 1) / out_h;
        const int ws = (ox * in_w) / out_w;
        const int we = ((ox + 1) * in_w + out_w - 1) / out_w;
        const int kh = he - hs;
        const int kw = we - ws;
        int sum[3] = {0, 0, 0};
        for (int y = hs; y < he; ++y) {
            const uint8_t* row = view + (y * in_w + ws) * 3;
            for (int x = 0; x < kw; ++x) {
                sum[0] += row[x * 3 + 0];
                sum[1] += row[x * 3 + 1];
                sum[2] += row[x * 3 + 2];
            }
        }
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            // Integer window sums are exact in float32 (< 2^24).
            float avg = __fdiv_rn(__fdiv_rn(static_cast<float>(sum[c]), static_cast<float>(kh)),
                                  static_cast<float>(kw));
            float v = __fmul_rn(__fmul_rn(avg, 2.0f), inv255);
            v = __fadd_rn(v, -1.0f);
            out[c * plane + dst] = __float2bfloat16_rn(v);
        }
        return;
    }

    // mode == kImageWAMVaeResizePilBilinear
    const int rx = crop_left + ox;
    const int ry = crop_top + oy;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int v;
        if (v_bounds == nullptr) {
            v = pil_horizontal(view, in_w, ry, rx, c, h_bounds, h_coeffs, h_ksize);
        } else {
            const int ymin = v_bounds[ry * 2 + 0];
            const int ylen = v_bounds[ry * 2 + 1];
            const int* k = v_coeffs + ry * v_ksize;
            int acc = 1 << (kPilPrecisionBits - 1);
            for (int t = 0; t < ylen; ++t) {
                acc += pil_horizontal(view, in_w, ymin + t, rx, c, h_bounds, h_coeffs, h_ksize) * k[t];
            }
            v = pil_clip8(acc);
        }
        out[c * plane + dst] = lut[v];
    }
}

}  // namespace

int imagewam_vae_preprocess_bf16(
    const uint8_t* view, const __nv_bfloat16* lut, __nv_bfloat16* out,
    int in_h, int in_w, int out_h, int out_w, int out_total_w, int col_offset,
    int mode,
    const int* h_bounds, const int* h_coeffs, int h_ksize, int crop_left,
    const int* v_bounds, const int* v_coeffs, int v_ksize, int crop_top,
    float inv255, cudaStream_t stream)
{
    if (view == nullptr || out == nullptr) return -1;
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) return -2;
    if (col_offset < 0 || col_offset + out_w > out_total_w) return -3;
    if (mode == kImageWAMVaeResizeIdentity) {
        if (lut == nullptr) return -4;
        if (in_h != out_h || in_w != out_w) return -5;
    } else if (mode == kImageWAMVaeResizeArea) {
        // no tables needed
    } else if (mode == kImageWAMVaeResizePilBilinear) {
        if (lut == nullptr) return -4;
        if ((h_bounds == nullptr) != (h_coeffs == nullptr)) return -6;
        if ((v_bounds == nullptr) != (v_coeffs == nullptr)) return -6;
        if (h_bounds == nullptr && crop_left + out_w > in_w) return -7;
        if (v_bounds == nullptr && crop_top + out_h > in_h) return -7;
        if (crop_left < 0 || crop_top < 0) return -7;
    } else {
        return -8;
    }
    const int total = out_h * out_w;
    const int blocks = (total + kThreads - 1) / kThreads;
    imagewam_vae_preprocess_kernel<<<blocks, kThreads, 0, stream>>>(
        view, lut, out, in_h, in_w, out_h, out_w, out_total_w, col_offset, mode,
        h_bounds, h_coeffs, h_ksize, crop_left, v_bounds, v_coeffs, v_ksize, crop_top, inv255);
    return static_cast<int>(cudaGetLastError());
}

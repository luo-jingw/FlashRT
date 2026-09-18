// ================================================================
// FlashRT — ImageWAM VAE input preprocessing
//
// One camera view, (in_h, in_w, 3) uint8 HWC, -> its column block of
// the VAE input image, (3, out_h, out_total_w) BF16 NCHW (batch 1),
// normalized to x*2/255-1. One launch per view; views are
// concatenated horizontally through `col_offset`.
//
// Modes (see flash_rt/models/imagewam/vae_preprocess.py):
//   0 identity     : in == out size; out = lut[v]
//   1 area         : torch F.interpolate(mode="area") semantics,
//                    avg = sum / kh / kw (float32, two rounded divs),
//                    then ((avg * 2) * inv255) - 1, rounded to BF16
//   2 pil_bilinear : PIL Image.resize(BILINEAR) fixed-point two-pass
//                    resample (22-bit coefficients, horizontal pass then
//                    vertical pass, uint8 rounding after each), center
//                    crop through crop_left/crop_top, then out = lut[v]
// ================================================================
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int kImageWAMVaeResizeIdentity = 0;
constexpr int kImageWAMVaeResizeArea = 1;
constexpr int kImageWAMVaeResizePilBilinear = 2;

// Returns 0 on success, a negative code for invalid arguments, or the
// positive cudaError_t of the launch.
//
// h_bounds/h_coeffs: horizontal-pass tables indexed by resized column,
// (resized_w, 2) int32 {xmin, xlen} and (resized_w, h_ksize) int32, or
// nullptr when the resized width equals in_w (no horizontal pass).
// v_bounds/v_coeffs: the same for rows, nullptr when resized height
// equals in_h. Only read in mode 2.
int imagewam_vae_preprocess_bf16(
    const uint8_t* view, const __nv_bfloat16* lut, __nv_bfloat16* out,
    int in_h, int in_w, int out_h, int out_w, int out_total_w, int col_offset,
    int mode,
    const int* h_bounds, const int* h_coeffs, int h_ksize, int crop_left,
    const int* v_bounds, const int* v_coeffs, int v_ksize, int crop_top,
    float inv255, cudaStream_t stream);

// ================================================================
// FlashRT — ImageWAM VAE ResnetBlock tail, NHWC BF16.
//
// y = bf16( r + bf16(h + h_bias) ),  r = res_bias ? bf16(res + res_bias) : res
//
// This is torch's conv2 bias add, optional nin_shortcut bias add and the
// residual `x + h` of flux2.autoencoder.ResnetBlock, with the same BF16
// rounding after each op, in one vectorized (8 x BF16) pass over
// bias-free convolution outputs.
// ================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// h, res, y: (rows, C) BF16 contiguous, 16-byte aligned; y may alias h
// or res. h_bias: (C,) BF16. res_bias: (C,) BF16 or nullptr. C % 8 == 0.
// Returns 0, a negative code for invalid arguments, or the positive
// cudaError_t of the launch.
int imagewam_bias_residual_nhwc_bf16(
    const __nv_bfloat16* h, const __nv_bfloat16* h_bias,
    const __nv_bfloat16* res, const __nv_bfloat16* res_bias,
    __nv_bfloat16* y, long long rows, int C, cudaStream_t stream);

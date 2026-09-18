// ================================================================
// FlashRT — ImageWAM VAE GroupNorm (+ optional SiLU), NHWC BF16.
//
// y[n, p, c] = x[n, p, c] * a[n, c] + b[n, c]           (float, FMA)
//   a = rstd[n, g] * gamma[c],  b = beta[c] - mean[n, g] * a,  g = c / (C/G)
// then, with apply_silu, the FLUX.2 `swish(x) = x * sigmoid(x)` chain
// with BF16 rounding after the norm, after the sigmoid and after the
// product, the same rounding points torch's three separate ops have.
//
// Optional `bias` (C,) BF16: the input is taken as bf16(x + bias), which
// is torch's separate conv-bias add (with its BF16 rounding) folded into
// the GroupNorm reads; the biased tensor is never written.
//
// Three launches: per-block Welford partial statistics, a per-(n, g)
// finalize (Chan merge) that writes the per-channel a/b, and a
// vectorized (8 x BF16) apply.
//
// Supported shapes: C % 8 == 0, C / 8 divides 256, C % G == 0, and
// C / G == 4 or C / G % 8 == 0. Covers the FLUX.2 AutoEncoder encoder
// (C in {128, 256, 512}, G = 32).
// ================================================================
#pragma once

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Bytes of float workspace imagewam_groupnorm_nhwc_bf16 needs for a
// (N, HW, C) input with G groups; 0 for an unsupported shape.
size_t imagewam_groupnorm_nhwc_workspace_bytes(int N, int HW, int C, int G);

// x, y: (N, HW, C) BF16 contiguous (NHWC, HW = H*W); y may alias x.
// bias: (C,) BF16 or nullptr. gamma, beta: (C,) BF16. Returns 0 on
// success, a negative code for an unsupported shape, misaligned pointer
// or too-small workspace, or the positive cudaError_t of the launches.
int imagewam_groupnorm_nhwc_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias, const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* y, void* workspace, size_t workspace_bytes,
    int N, int HW, int C, int G, float eps, int apply_silu, cudaStream_t stream);

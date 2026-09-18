// ================================================================
// FlashRT — Activation kernel declarations
// GeLU, SiLU, Gate*Act*Mul (BF16/FP16 and fused FP8 variants)
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

// NOTE: these were previously misnamed `gate_silu_mul*`; they compute the
// tanh/sigmoid-approx GELU (GeGLU), not SiLU. Renamed to `gate_geglu*`.
void gate_geglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream = 0);

void gelu_inplace(__nv_bfloat16* x, int n, cudaStream_t stream = 0);
void relu2_inplace_bf16(__nv_bfloat16* x, int n, cudaStream_t stream = 0);

// G7.11 — fused (bias add + GELU(tanh)) in-place on bf16 tensor.
// x: (M, N) bf16; bias: (N,) bf16 broadcast over rows. Replaces
// add_bias_bf16 + gelu_inplace pair (2 launches -> 1).
void bias_gelu_inplace_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                              int M, int N, cudaStream_t stream = 0);

// Strict variant matching add_bias_bf16 + gelu_inplace numerics: the
// bias-add result is rounded back to BF16 before applying GELU.
void bias_gelu_inplace_bf16_strict(__nv_bfloat16* x,
                                   const __nv_bfloat16* bias,
                                   int M, int N, cudaStream_t stream = 0);

void gate_geglu_merged(const __nv_bfloat16* merged, __nv_bfloat16* out,
                           int seq, int half_dim, cudaStream_t stream = 0);

void gate_geglu_merged_fp8(const __nv_bfloat16* merged, __nv_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_geglu_fp16(const __half* gate, const __half* up,
                        __half* out, int n, cudaStream_t stream = 0);

void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream = 0);

void gate_geglu_merged_fp16(const __half* merged, __half* out,
                                int seq, int half_dim, cudaStream_t stream = 0);

// ImageWAM/FLUX.2 real MLP gate (opportunities.md OPT-002 follow-up):
// same "merged, strided-halves" layout as gate_geglu_merged_fp16 above
// (merged[row, col] = gate, merged[row, half_dim+col] = up), but SiLU
// instead of GELU -- matches the real `SiLUActivation.forward`
// (`x1, x2 = x.chunk(2, dim=-1); return silu(x1) * x2`) exactly, found
// while reading flux2/model.py directly. The existing gate_geglu_*
// kernels use GELU and are for a DIFFERENT model; not reusable here.
// `row_stride`: 0 (default) means "tightly packed, use half_dim*2" --
// every existing caller's own layout, unchanged. Pass a wider buffer's
// real row width to read gate/up from a column-slice of it instead
// (opportunities.md op-fusion audit finding 1).
// `out_row_stride`: 0 (default) means a packed `(seq, half_dim)` output.
// Pass a wider output buffer's row width to write into a column slice of
// it (the merged single-stream `linear2` input, roadmap item 4).
void silu_glu_merged_fp16(const __half* merged, __half* out,
                           int seq, int half_dim, cudaStream_t stream = 0,
                           int row_stride = 0, int out_row_stride = 0);

// Element-wise multiply: out[i] = a[i] * b[i] for i in [0, n).
// FP16 inputs and output, FP32 multiply.  Used by R3.1 split-G7 path
// to combine GELU(gate) with up after two separate GEMMs.
void mul_fp16(const __half* a, const __half* b, __half* out,
              int n, cudaStream_t stream = 0);

void gate_geglu_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                                    int seq, int half_dim,
                                    const float* d_scale, cudaStream_t stream = 0);

// Split SiLU: separate gate and up buffers → FP8 output
// Matches pi05 silu_mul_split_fp8_k (split gate+up GEMMs for L2 optimization)
void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                              __nv_fp8_e4m3* out, int n,
                              const float* d_scale, cudaStream_t stream = 0);

// GeGLU with fused per-tensor amax: writes fp16 output and folds its
// abs-max into a caller-zeroed device accumulator (for fused dynamic
// FP8 quantize). d_amax must be memset to 0 by the caller first.
void gate_geglu_amax_fp16(const __half* gate, const __half* up, __half* out,
                          float* d_amax, int n, cudaStream_t stream = 0);

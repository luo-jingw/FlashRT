// ================================================================
// FlashRT — Cross-layer fusion kernel declarations
// Fused gate*residual + AdaRMSNorm -> FP8
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signature, backward compatible) ──

void gate_residual_ada_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variant ──

void gate_residual_ada_norm_fp8_fp16(__half* residual, const __half* x,
                                      const __half* gate, const __half* weight,
                                      const __half* style,
                                      __nv_fp8_e4m3* out, __half* gate_out,
                                      int seq_len, int dim, float eps,
                                      const float* d_scale, cudaStream_t stream = 0);

void gate_residual_ada_norm_fp16(__half* residual, const __half* x,
                                  const __half* gate, const __half* weight,
                                  const __half* style,
                                  __half* out, __half* gate_out,
                                  int seq_len, int dim, float eps,
                                  cudaStream_t stream = 0);

void gate_residual_ada_norm_int8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 int8_t* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 float* d_scales, cudaStream_t stream = 0);

// ── Fused gated residual + next AdaLayerNorm (ImageWAM / FLUX.2 DiT) ──
// residual[r,c] += fp16(gate[c]) * proj[r,c]   (stored in the residual dtype)
// out[r,:] = fp16(LN_no_affine(residual[r,:]) * (1 + fp16(scale)) + fp16(shift))
// gate/scale/shift: (dim,) FP32, rounded to FP16 in-kernel. Bit-identical to
// gate_res_{bf16res,fp16} followed by ada_layer_norm_{bf16in_fp16out,fp16}
// with FP16 modulation vectors. `dim` even and `rows` > 0 (otherwise
// std::invalid_argument). `out == nullptr`: residual update only (no
// AdaLN follows); `scale`/`shift` are then not read.

void gate_res_ada_layer_norm_bf16res(const __half* proj, const float* gate,
                                     __nv_bfloat16* residual,
                                     const float* scale, const float* shift, __half* out,
                                     int rows, int dim, float eps, cudaStream_t stream = 0);

void gate_res_ada_layer_norm_fp16(const __half* proj, const float* gate,
                                  __half* residual,
                                  const float* scale, const float* shift, __half* out,
                                  int rows, int dim, float eps, cudaStream_t stream = 0);

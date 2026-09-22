// ============================================================================
//  FlashRT — fused (gated residual + AdaLayerNorm-no-affine + NVFP4 quantize
//  + CUTLASS SFA write) kernel, for ImageWAM's own producer/consumer pair.
//
//  Standalone, additive kernel. NOT wired into the production pipeline or
//  the main CMake build — see tests/test_fused_norm_fp4_kernel.py.
//
//  Replaces the two-kernel real pipeline sequence
//      fvk.gate_res_bf16res(gemm_out, gate, residual, n)            // decoder_fused.cu
//      fvk.ada_layer_norm_bf16in_fp16out(residual, scale, shift,    // norm.cu
//                                         modded_fp16, seq, dim, eps)
//      fvk.quantize_fp4_dynamic_sfa_fp16(modded_fp16, packed, sfa,  // quantize_fp4_sfa.cu
//                                         seq, dim, /*is_sfb=*/false)
//  (BF16-residual variant; the `_fp16res` entry point below matches the
//  FP16-residual sequence gate_res_fp16 + ada_layer_norm_fp16 +
//  quantize_fp4_dynamic_sfa_fp16, used by ImageWAM's ActionDiT layers)
//  with ONE kernel launch that writes NVFP4-packed bytes + per-16-block SFA
//  scale factors directly, never materializing the intermediate fp16
//  `modded` buffer.
//
//  `gate` is passed as a per-row-broadcast [dim] vector (read once per row
//  inside the kernel), NOT the [seq,dim] broadcast-materialized copy
//  `gate_res_*_kernel`'s own flat indexing required -- fusing the residual
//  update into a one-CTA-per-row kernel removes the need for that
//  materialize entirely (a real memory-traffic win, not just launch-count).
//
//  `inv_s` (nullable): optional per-channel AWQ input-scale fold, multiplied
//  into the modulated value immediately before the fp16-round + FP4-quantize
//  step -- same established convention as
//  csrc/fused_fp4/layer_norm_fp4_sfa.cu's own `inv_s` parameter. Pass
//  nullptr for the plain (no AWQ) path.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_norm_fp4 {

// BF16 residual (ImageWAM backbone double/single-stream layers: hidden=3072).
int gate_res_ada_layer_norm_fp4_sfa_bf16res(
    void* residual_bf16,        // [seq, dim] BF16, updated in place
    const void* gemm_out_fp16,  // [seq, dim] FP16 (proj/mlp GEMM output)
    const void* gate_fp16,      // [dim] FP16 (per-layer AdaLN gate, broadcast per row)
    const void* scale_fp16,     // [dim] FP16 (next AdaLN scale)
    const void* shift_fp16,     // [dim] FP16 (next AdaLN shift)
    const void* inv_s_fp16,     // [dim] FP16 or nullptr (AWQ fold, optional)
    void* dst_packed,           // [seq, dim/2] uint8 (e2m1 nibble-packed)
    void* dst_sfa,              // CUTLASS SFA tile-interleaved UE4M3 bytes
    int seq_len, int dim, float eps, cudaStream_t stream);

// FP16 residual (ImageWAM ActionDiT double/single layers: action_hidden_dim
// or action_attn_width).
int gate_res_ada_layer_norm_fp4_sfa_fp16res(
    void* residual_fp16,
    const void* gemm_out_fp16,
    const void* gate_fp16,
    const void* scale_fp16,
    const void* shift_fp16,
    const void* inv_s_fp16,
    void* dst_packed,
    void* dst_sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

}  // namespace fused_norm_fp4
}  // namespace flash_rt

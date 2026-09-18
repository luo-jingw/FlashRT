// Declarations of the csrc operations the ImageWAM native pipeline launches
// that have no csrc header of their own. Signatures match the definitions
// in csrc/kernels/norm.cu and csrc/kernels/elementwise.cu (the same
// declarations csrc/bindings.cpp uses). The other operations come from
// their csrc headers (norm.cuh, rope.cuh, activation.cuh, decoder_fused.cuh,
// elementwise.cuh, attention_cublas.cuh).
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_CSRC_OPERATIONS_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_CSRC_OPERATIONS_H

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

void ada_layer_norm_fp16(const __half* x, const __half* scale, const __half* shift, __half* out,
                         int seq_len, int dim, float eps, cudaStream_t stream);
void ada_layer_norm_bf16in_fp16out(const __nv_bfloat16* x, const __half* scale,
                                   const __half* shift, __half* out, int seq_len, int dim,
                                   float eps, cudaStream_t stream);
void gpu_cast_fp32_to_fp16(const float* src, __half* dst, int n, cudaStream_t stream);
void gpu_euler_step(float* actions, const __half* velocity, int T, int action_dim, float dt,
                    int vel_elem_offset, cudaStream_t stream);
void gpu_strided_copy_fp16(const __half* src, __half* dst, int rows, int dst_cols,
                           int src_stride, int col_offset, cudaStream_t stream);

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_CSRC_OPERATIONS_H

// NVFP4 linear of the native pipeline: Nvfp4Linear in
// flash_rt/models/imagewam/quant_linear.py (dynamic per-16 activation
// quantization into the handed-off scratch, then the CUTLASS NVFP4 GEMM
// variant the setup producer picked). Available only in builds with the
// flash_rt_fp4 kernels (SM100-class, FLASHRT_IMAGEWAM_NATIVE_NVFP4).
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_FP4_LINEAR_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_FP4_LINEAR_H

#include "flashrt/cpp/models/imagewam/c_api.h"

#include <cuda_runtime_api.h>

namespace flashrt {
namespace models {
namespace imagewam {

bool nvfp4_available();

// out (m, n) fp16 = x (m, k) fp16 @ W^T; throws std::runtime_error on failure.
void nvfp4_linear(const frt_imagewam_linear& linear, const void* x, void* out, int m,
                  cudaStream_t stream);

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_FP4_LINEAR_H

#include "fp4_linear.h"

#include "flashrt/cpp/models/imagewam/c_api.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#ifdef FLASHRT_IMAGEWAM_NATIVE_NVFP4
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#endif

namespace flashrt {
namespace models {
namespace imagewam {

#ifdef FLASHRT_IMAGEWAM_NATIVE_NVFP4

bool nvfp4_available() { return true; }

void nvfp4_linear(const frt_imagewam_linear& l, const void* x, void* out, int m,
                  cudaStream_t stream) {
    int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(x, l.act_packed, l.act_scales, m, l.k,
                                                          false, stream);
    if (rc != 0) {
        throw std::runtime_error("quantize_fp4_dynamic_sfa_fp16 failed rc=" + std::to_string(rc));
    }
    rc = flash_rt::fp4::cutlass_fp4_gemm_variant(l.fp4_variant, l.act_packed, l.act_scales,
                                                  l.weight, l.weight_scales, out, m, l.n, l.k,
                                                  1.0f, 0.0f, stream);
    if (rc != 0) {
        throw std::runtime_error("cutlass_fp4_gemm_variant(" + std::to_string(l.fp4_variant) +
                                 ") failed rc=" + std::to_string(rc));
    }
}

#else

bool nvfp4_available() { return false; }

void nvfp4_linear(const frt_imagewam_linear&, const void*, void*, int, cudaStream_t) {
    throw std::runtime_error("NVFP4 linear needs a build with the flash_rt_fp4 kernels (SM100-class)");
}

#endif

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt
